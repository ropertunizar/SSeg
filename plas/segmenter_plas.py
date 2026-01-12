import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from skimage.segmentation import find_boundaries
from PIL import Image
from torchvision import transforms


class SuperpixelLabelExpander:
    def __init__(self, device, seed: int | None = None, deterministic: bool = True):
        """
        Superpixel label propagation (PLAS) wrapper.

        Args:
            device: 'cuda' or 'cpu'.
            seed: Explicit integer seed for reproducibility. If None, defaults to 42.
            deterministic: If True attempt to remove remaining nondeterminism.

        Note: For full reproducibility, set os.environ['GLOBAL_SEED'] in your main script and always pass the same seed to all modules.
        """
        self.device = device
        if seed is None:
            seed = 42
        self.seed = int(seed)
        self.deterministic = deterministic
        self.deterministic_parent = self.deterministic
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(self.seed)
        # Make cudnn deterministic if requested
        if deterministic and torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = False  # type: ignore
                if hasattr(torch.backends.cudnn, 'deterministic'):
                    torch.backends.cudnn.deterministic = True  # type: ignore
                # Disable TF32 for tighter reproducibility
                # Re-enable TF32 for performance (still keeping other deterministic aspects)
                if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True  # type: ignore
            except Exception:
                pass

    def expand_labels(self, image, points_gt, labels_gt, unlabeled=0, features_sam2=None, **kwargs):
        # Reinforce deterministic backend settings each call (some external code may flip them)
        if self.deterministic and torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = False  # type: ignore
            except Exception:
                pass

        sigma_xy = 0.631
        sigma_cnn = 0.5534
        alpha = 1140

        def members_from_clusters(sigma_val_xy, sigma_val_cnn, XY_features, CNN_features, clusters):
            B, K, _ = clusters.shape
            sigma_array_xy = torch.full((B, K), sigma_val_xy, device=self.device)
            sigma_array_cnn = torch.full((B, K), sigma_val_cnn, device=self.device)
            
            clusters_xy = clusters[:,:,0:2]
            dist_sq_xy = torch.cdist(XY_features, clusters_xy)**2

            clusters_cnn = clusters[:,:,2:]
            dist_sq_cnn = torch.cdist(CNN_features, clusters_cnn)**2

            soft_memberships = F.softmax( (- dist_sq_xy / (2.0 * sigma_array_xy**2)) + (- dist_sq_cnn / (2.0 * sigma_array_cnn**2)) , dim = 2)                # shape = [B, N, K] 
            
            return soft_memberships

        def enforce_connectivity(hard, H, W, K_max, connectivity = True):
            # INPUTS
            # 1. posteriors:    shape = [B, N, K]
            B = 1

            hard_assoc = torch.unsqueeze(hard, 0).detach().cpu().numpy()                                 # shape = [B, N]
            hard_assoc_hw = hard_assoc.reshape((B, H, W))    

            segment_size = (H * W) / (int(K_max) * 1.0)

            min_size = int(0.06 * segment_size)
            max_size = int(H*W*10)

            hard_assoc_hw = hard_assoc.reshape((B, H, W))
            
            for b in range(hard_assoc.shape[0]):
                if connectivity:
                    spix_index_connect = _enforce_label_connectivity_cython(hard_assoc_hw[None, b, :, :], min_size, max_size, 0)[0]
                else:
                    spix_index_connect = hard_assoc_hw[b,:,:]

            return spix_index_connect

        class CustomLoss(nn.Module):
            def __init__(self, device, clusters_init, N, XY_features, CNN_features, features_cat, labels, sigma_val_xy = 0.5, sigma_val_cnn = 0.5, alpha = 1, num_pixels_used = 1000, gen: torch.Generator | None = None):
                super(CustomLoss, self).__init__()
                self.alpha = alpha # Weighting for the distortion loss
                self.clusters=nn.Parameter(clusters_init, requires_grad=True)   # clusters (torch.FloatTensor: shape = [B, K, C])
                B, K, _ = self.clusters.shape

                self.N = N

                self.sigma_val_xy = sigma_val_xy
                self.sigma_val_cnn = sigma_val_cnn
                self.device = device

                self.sigma_array_xy = torch.full((B, K), self.sigma_val_xy, device=self.device)
                self.sigma_array_cnn = torch.full((B, K), self.sigma_val_cnn, device=self.device)

                self.XY_features = XY_features
                self.CNN_features = CNN_features
                self.features_cat = features_cat

                self.labels = labels
                self.num_pixels_used = num_pixels_used
                self._gen = gen  # optional dedicated generator
                # Precompute a fixed subset of pixel indices once for all iterations for reproducibility
                use_n = min(self.N, self.num_pixels_used)
                dev = self.XY_features.device
                if self._gen is not None and use_n < self.N:
                    self.sample_indexes = torch.randperm(self.N, generator=self._gen, device=dev)[:use_n]
                elif use_n < self.N:
                    self.sample_indexes = torch.randperm(self.N, device=dev)[:use_n]
                else:
                    self.sample_indexes = torch.arange(self.N, device=dev)


            def forward(self):
                # computes the distortion loss of the superpixels and also our novel conflict loss
                #
                # INPUTS:
                # 1) features:      (torch.FloatTensor: shape = [B, N, C]) defines for each image the set of pixel features

                # B is the batch dimension
                # N is the number of pixels
                # K is the number of superpixels

                # RETURNS:
                # 1) sum of distortion loss and conflict loss scaled by alpha (we use lambda in the paper but this means something else when coding)
                # Use precomputed fixed subset
                indexes = self.sample_indexes

                ##################################### DISTORTION LOSS #################################################
                # Calculate the distance between pixels and superpixel centres by expanding our equation: (a-b)^2 = a^2-2ab+b^2 
                features_cat_select = self.features_cat[:,indexes,:]
                dist_sq_cat = torch.cdist(features_cat_select, self.clusters)**2

                # XY COMPONENT
                clusters_xy = self.clusters[:,:,0:2]

                XY_features_select = self.XY_features[:,indexes,:]
                dist_sq_xy = torch.cdist(XY_features_select, clusters_xy)**2

                # CNN COMPONENT
                clusters_cnn = self.clusters[:,:,2:]

                CNN_features_select = self.CNN_features[:,indexes,:]
                dist_sq_cnn = torch.cdist(CNN_features_select, clusters_cnn)**2              

                B, K, _ = self.clusters.shape
                
                soft_memberships = F.softmax( (- dist_sq_xy / (2.0 * self.sigma_array_xy**2)) + (- dist_sq_cnn / (2.0 * self.sigma_array_cnn**2)) , dim = 2)                # shape = [B, N, K]  

                # The distances are weighted by the soft memberships
                dist_sq_weighted = soft_memberships * dist_sq_cat                                           # shape = [B, N, K] 

                distortion_loss = torch.mean(dist_sq_weighted)                                          # shape = [1]

                ###################################### CONFLICT LOSS ###################################################
                # print("labels", labels.shape)                                                         # shape = [B, 1, H, W]
                
                labels_reshape = self.labels.permute(0,2,3,1).float()                                   # shape = [B, H, W, 1]   

                # Find the indexes of the class labels larger than 0 (0 is means unknown class)
                label_locations = torch.gt(labels_reshape, 0).float()                                   # shape = [B, H, W, 1]
                label_locations_flat = torch.flatten(label_locations, start_dim=1, end_dim=2)           # shape = [B, N, 1]  

                XY_features_label = (self.XY_features * label_locations_flat)[0]                        # shape = [N, 2]
                non_zero_indexes = torch.abs(XY_features_label).sum(dim=1) > 0                          # shape = [N] 
                XY_features_label_filtered = XY_features_label[non_zero_indexes].unsqueeze(0)           # shape = [1, N_labelled, 2]
                dist_sq_xy = torch.cdist(XY_features_label_filtered, clusters_xy)**2                    # shape = [1, N_labelled, K]

                CNN_features_label = (self.CNN_features * label_locations_flat)[0]                      # shape = [N, 15]
                CNN_features_label_filtered = CNN_features_label[non_zero_indexes].unsqueeze(0)         # shape = [1, N_labelled, 15]
                dist_sq_cnn = torch.cdist(CNN_features_label_filtered, clusters_cnn)**2                 # shape = [1, N_labelled, K]

                soft_memberships = F.softmax( (- dist_sq_xy / (2.0 * self.sigma_array_xy**2)) + (- dist_sq_cnn / (2.0 * self.sigma_array_cnn**2)) , dim = 2)          # shape = [B, N_labelled, K]  
                soft_memberships_T = torch.transpose(soft_memberships, 1, 2)                            # shape = [1, K, N_labelled]

                labels_flatten = torch.flatten(labels_reshape, start_dim=1, end_dim=2)[0]               # shape = [N, 1]
                labels_filtered = labels_flatten[non_zero_indexes].unsqueeze(0)                         # shape = [1, N_labelled, 1] 

                # Use batched matrix multiplication to find the inner product between all of the pixels 
                innerproducts = torch.bmm(soft_memberships, soft_memberships_T)                         # shape = [1, N_labelled, N_labelled]

                # Create an array of 0's and 1's based on whether the class of both the pixels are equal or not
                # If they are the the same class, then we want a 0 because we don't want to add to the loss
                # If the two pixels are not the same class, then we want a 1 because we want to penalise this
                check_conflicts_binary = (~torch.eq(labels_filtered, torch.transpose(labels_filtered, 1, 2))).float()      # shape = [1, N_labelled, N_labelled]

                # Multiply these ones and zeros with the innerproduct array
                # Only innerproducts for pixels with conflicting labels will remain
                conflicting_innerproducts = torch.mul(innerproducts, check_conflicts_binary)           # shape = [1, N_labelled, N_labelled]

                # Find average of the remaining values for the innerproducts 
                # If we are using batches, then we add this value to our previous stored value for the points loss
                conflict_loss = torch.mean(conflicting_innerproducts)                                # shape = [1]

                return distortion_loss + self.alpha*conflict_loss, distortion_loss, self.alpha*conflict_loss

        def optimize_spix(criterion, optimizer, norm_val_x, norm_val_y, image_width, image_height, num_iterations=1000):
            best_clusters = criterion.clusters
            prev_loss = float("inf")
            for i in range(1, num_iterations):
                loss, distortion_loss, conflict_loss = criterion()
                if i % 10 == 0:
                    with torch.no_grad():
                        cx = torch.unsqueeze(torch.clamp(criterion.clusters[0, :, 0], 0, ((image_width - 1) * norm_val_x)), dim=1)
                        cy = torch.unsqueeze(torch.clamp(criterion.clusters[0, :, 1], 0, ((image_height - 1) * norm_val_y)), dim=1)
                        ctemp = torch.unsqueeze(torch.cat((cx, cy, criterion.clusters[0, :, 2:]), dim=1), dim=0)
                    criterion.clusters.data.copy_(ctemp)
                if loss < prev_loss:
                    best_clusters = criterion.clusters
                    prev_loss = loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            return best_clusters

        def prop_to_unlabelled_spix_feat(sparse_labels, connected, features_cnn, H, W):
            # Detach and prepare CNN features
            features_cnn = features_cnn.detach().cpu().numpy()[0]  # shape = [N, C]
            features_cnn_reshape = np.reshape(features_cnn, (H, W, features_cnn.shape[1]))  # shape = [H, W, C]

            # Calculate unique superpixels and initialize feature array
            unique_spix = np.unique(connected)
            spix_features = np.zeros((len(unique_spix), features_cnn.shape[1] + 1))

            # Calculate average features for each superpixel
            for i, spix in enumerate(unique_spix):
                r, c = np.where(connected == spix)
                features_curr_spix = features_cnn_reshape[r, c]
                spix_features[i, 0] = spix  # store spix index
                spix_features[i, 1:] = np.mean(features_curr_spix, axis=0)  # store average feature vector

            # Label array for all labeled pixels
            mask_np = np.array(sparse_labels).squeeze()
            # If mask_np is 1-D (no labels), make sure downstream code handles it
            if mask_np.ndim == 0:
                mask_np = np.expand_dims(mask_np, 0)
            labelled_indices = np.argwhere(mask_np > 0)
            labels = []
            for yx in labelled_indices:
                # labelled_indices entries may be (y,x) pairs
                if yx.size == 1:
                    # single index case (unexpected), skip
                    continue
                y, x = int(yx[0]), int(yx[1])
                labels.append([int(mask_np[y, x]) - 1, int(connected[y, x]), y, x])

            labels_array = np.array(labels)
            # Ensure labels_array is 2D with shape (N,4) even if empty
            if labels_array.size == 0:
                labels_array = np.zeros((0, 4), dtype=int)

            # Calculate labels for each superpixel with points in it
            spix_labels = []
            for spix in unique_spix:
                # Only attempt to index labels_array if it has rows
                if labels_array.shape[0] > 0 and spix in labels_array[:, 1]:
                    label_indices = np.where(labels_array[:, 1] == spix)[0]
                    labels_vals = labels_array[label_indices, 0].astype(int)
                    if labels_vals.size == 0:
                        continue
                    most_common = np.bincount(labels_vals).argmax()
                    spix_features_row = spix_features[unique_spix == spix, 1:].flatten()
                    spix_labels.append([int(spix), int(most_common)] + list(spix_features_row))

            # Convert spix_labels to array; ensure 2D even if empty
            if len(spix_labels) == 0:
                spix_labels = np.zeros((0, 2), dtype=int)
            else:
                spix_labels = np.array(spix_labels)

            # Prepare empty mask and propagate labels
            prop_mask = np.full((H, W), np.nan)

            for i, spix in enumerate(unique_spix):
                r, c = np.where(connected == spix)

                # If already labeled, use label from spix_labels
                if spix_labels.size and spix in spix_labels[:, 0]:
                    label = int(spix_labels[spix_labels[:, 0] == spix, 1][0])
                    prop_mask[r, c] = label
                else:
                    # Find the nearest labeled superpixel by features
                    if spix_labels.size == 0:
                        # No labeled superpixels available; set to 0
                        prop_mask[r, c] = 0
                    else:
                        labeled_spix_features = spix_labels[:, 2:].astype(float)
                        one_spix_features = spix_features[i, 1:].astype(float)
                        # If labeled_spix_features has zero columns (degenerate), fallback
                        if labeled_spix_features.size == 0:
                            prop_mask[r, c] = 0
                        else:
                            distances = np.linalg.norm(labeled_spix_features - one_spix_features, axis=1)
                            nearest_spix_idx = np.argmin(distances)
                            nearest_label = int(spix_labels[nearest_spix_idx, 1])
                            prop_mask[r, c] = nearest_label

            return prop_mask

        def generate_segmented_image(image, num_labels, image_height, image_width, num_classes, unlabeled, points_gt, labels_gt, ensemble=True):
            # Load necessary modules and functions
            from .spixel_utils import xylab, find_mean_std, img2lab, ToTensor, compute_init_spixel_feat, get_spixel_init
            from .ssn import CNN
            from torch.optim import Adam, lr_scheduler

            # Initialize variables: respect parent expander device to avoid mixing cpu/cuda
            device = self.device if hasattr(self, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
            k = 100
            norm_val_x = 10 / image_width
            norm_val_y = 10 / image_height
            learning_rate = 0.1
            num_iterations = 50
            num_pixels_used = 3000

            sparse_coords = np.zeros((image_height, image_width), dtype=int)
                
            # Populate sparse_coords with labels at the specified points
            for p, l in zip(points_gt, labels_gt):
                sparse_coords[p[1], p[0]] = l + 1

            # Prepare sparse labels
            sparse_mask = sparse_coords

            # Create sparse labels tensor from the sparse mask
            sparse_mask = np.expand_dims(sparse_mask, axis=0)
            sparse_labels = torch.from_numpy(sparse_mask).to(device)

            # Expand dimensions to match the expected input format
            sparse_labels = torch.unsqueeze(sparse_labels, 0).to(device)

            # Standardize image
            means, stds = find_mean_std(image)
            image = (image - means) / stds
            transform = transforms.Compose([img2lab(), ToTensor()])
            img_lab = transform(image)
            img_lab = torch.unsqueeze(img_lab, 0)

            # Obtain features
            xylab_function = xylab(1.0, norm_val_x, norm_val_y)
            CNN_function = CNN(5, 64, 100)
            model_dict = CNN_function.state_dict()
            
            # Get the path to the checkpoint file relative to this module
            import os
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level from plas/
            ckp_path = os.path.join(current_dir, "checkpoints", "standardization_C=100_step70000.pth")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            obj = torch.load(ckp_path, map_location=device)
            pretrained_dict = obj['net']
            pretrained_dict = {key[4:]: val for key, val in pretrained_dict.items() if key[4:] in model_dict}
            model_dict.update(pretrained_dict)
            CNN_function.load_state_dict(pretrained_dict)
            CNN_function.to(device)
            CNN_function.eval()

            spixel_centres = get_spixel_init(k, image_width, image_height)
            XYLab, X, Y, Lab = xylab_function(img_lab)
            XYLab = XYLab.to(device)
            X = X.to(device)
            Y = Y.to(device)

            with torch.no_grad():
                features = CNN_function(XYLab)
            
            # change dtype of features to float32
            features = features.float()
            features_magnitude_mean = torch.mean(torch.norm(features, p=2, dim=1))
            features_rescaled = (features / features_magnitude_mean)
            features_cat = torch.cat((X, Y, features_rescaled), dim=1)
            XY_cat = torch.cat((X, Y), dim=1)

            mean_init = compute_init_spixel_feat(features_cat, torch.from_numpy(spixel_centres[0].flatten()).long().to(device), k)

            CNN_features = torch.flatten(features_rescaled, start_dim=2, end_dim=3)
            CNN_features = torch.transpose(CNN_features, 2, 1)
            XY_features = torch.flatten(XY_cat, start_dim=2, end_dim=3)
            XY_features = torch.transpose(XY_features, 2, 1)
            features_cat = torch.flatten(features_cat, start_dim=2, end_dim=3)
            features_cat = torch.transpose(features_cat, 2, 1)

            # Respect deterministic mode from parent expander (we disable benchmark earlier if deterministic)
            if not (hasattr(self, 'deterministic_parent') and self.deterministic_parent):
                torch.backends.cudnn.benchmark = True

            if ensemble:
                # print("Ensemble")
                sigma_xy_1, sigma_cnn_1, alpha_1 = 0.5597, 0.5539, 1500
                sigma_xy_2, sigma_cnn_2, alpha_2 = 0.5309, 0.846, 1590
                sigma_xy_3, sigma_cnn_3, alpha_3 = 0.631, 0.5534, 1140

                criterion_1 = CustomLoss(self.device, mean_init, image_height * image_width, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy_1, sigma_val_cnn=sigma_cnn_1, alpha=alpha_1, num_pixels_used=num_pixels_used, gen=(self._gen if self.deterministic else None)).to(device)
                optimizer_1 = Adam(criterion_1.parameters(), lr=learning_rate)

                criterion_2 = CustomLoss(self.device, mean_init, image_height * image_width, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy_2, sigma_val_cnn=sigma_cnn_2, alpha=alpha_2, num_pixels_used=num_pixels_used, gen=(self._gen if self.deterministic else None)).to(device)
                optimizer_2 = Adam(criterion_2.parameters(), lr=learning_rate)

                criterion_3 = CustomLoss(self.device, mean_init, image_height * image_width, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy_3, sigma_val_cnn=sigma_cnn_3, alpha=alpha_3, num_pixels_used=num_pixels_used, gen=(self._gen if self.deterministic else None)).to(device)
                optimizer_3 = Adam(criterion_3.parameters(), lr=learning_rate)

                best_clusters_1 = optimize_spix(criterion_1, optimizer_1, norm_val_x, norm_val_y, image_width, image_height, num_iterations=num_iterations)
                best_members_1 = members_from_clusters(sigma_xy_1, sigma_cnn_1, XY_features, CNN_features, best_clusters_1)

                best_clusters_2 = optimize_spix(criterion_2, optimizer_2, norm_val_x, norm_val_y, image_width, image_height, num_iterations=num_iterations)
                best_members_2 = members_from_clusters(sigma_xy_2, sigma_cnn_2, XY_features, CNN_features, best_clusters_2)

                best_clusters_3 = optimize_spix(criterion_3, optimizer_3, norm_val_x, norm_val_y, image_width, image_height, num_iterations=num_iterations)
                best_members_3 = members_from_clusters(sigma_xy_3, sigma_cnn_3, XY_features, CNN_features, best_clusters_3)

                best_members_1_max = torch.squeeze(torch.argmax(best_members_1, 2))
                best_members_2_max = torch.squeeze(torch.argmax(best_members_2, 2))
                best_members_3_max = torch.squeeze(torch.argmax(best_members_3, 2))

                def overlay_boundaries_on_image(original_image, assignments, boundary_color=(1, 0, 0)):
                    """
                    Overlay red boundaries of superpixels on the original image.

                    Parameters:
                    - original_image: Original image as a NumPy array (H, W, 3), values in [0, 255].
                    - assignments: Superpixel assignments as a NumPy array (H, W).
                    - boundary_color: Tuple of RGB values for the boundary color, in [0, 1].

                    Returns:
                    - overlaid_image: Image with red superpixel boundaries overlaid.
                    """
                    assignments = assignments.cpu().numpy().reshape(original_image.shape[:2])
                    boundaries = find_boundaries(assignments, mode='outer')

                    # Normalize the original image to [0, 1] for blending
                    original_image = original_image.astype(float)
                    overlaid_image = original_image.copy()

                    # Apply the boundary color
                    for i, color in enumerate(boundary_color):
                        overlaid_image[..., i][boundaries] = color

                    # Convert back to [0, 255] for display
                    return (overlaid_image*255).astype(np.uint8)


                connected_1 = enforce_connectivity(best_members_1_max, image_height, image_width, k, connectivity=True)
                connected_2 = enforce_connectivity(best_members_2_max, image_height, image_width, k, connectivity=True)
                connected_3 = enforce_connectivity(best_members_3_max, image_height, image_width, k, connectivity=True)

                prop_1 = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected_1, CNN_features, image_height, image_width)
                prop_2 = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected_2, CNN_features, image_height, image_width)
                prop_3 = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected_3, CNN_features, image_height, image_width)

                prop_1_onehot = np.eye(num_classes, dtype=np.int32)[prop_1.astype(np.int32)]
                prop_2_onehot = np.eye(num_classes, dtype=np.int32)[prop_2.astype(np.int32)]
                prop_3_onehot = np.eye(num_classes, dtype=np.int32)[prop_3.astype(np.int32)]

                prop_count = prop_1_onehot + prop_2_onehot + prop_3_onehot

                if unlabeled == 0:
                    propagated_full = np.argmax(prop_count[:, :, 1:], axis=-1) + 1
                    propagated_full[prop_count[:, :, 0] == 3] = 0
                else:
                    propagated_full = np.argmax(prop_count[:, :, :-1], axis=-1)
                    propagated_full[prop_count[:, :, unlabeled] == 3] = unlabeled

            else:
                # print("Single")
                criterion = CustomLoss(self.device, mean_init, image_height * image_width, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy, sigma_val_cnn=sigma_cnn, alpha=alpha, num_pixels_used=num_pixels_used, gen=(self._gen if self.deterministic else None)).to(device)
                optimizer = Adam(criterion.parameters(), lr=learning_rate)
                best_clusters = optimize_spix(criterion, optimizer, norm_val_x, norm_val_y, image_width, image_height, num_iterations=num_iterations)
                best_members = members_from_clusters(sigma_xy, sigma_cnn, XY_features, CNN_features, best_clusters)
                connected = enforce_connectivity(torch.squeeze(torch.argmax(best_members, 2)), image_height, image_width, k, connectivity=True)
                propagated_full = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected, CNN_features, image_height, image_width)

            # print(f"Time taken by ensemble: {end_ensemble - start_ensemble} seconds")
            return propagated_full
        
        num_labels = len(labels_gt)
        image_width = image.shape[1]
        image_height = image.shape[0]
        num_classes = kwargs.get('num_classes', 35)  # Allow num_classes to be passed, default to 35

        expanded_image = generate_segmented_image(image, num_labels, image_height, image_width, num_classes, unlabeled, points_gt, labels_gt, ensemble=True)

        return expanded_image