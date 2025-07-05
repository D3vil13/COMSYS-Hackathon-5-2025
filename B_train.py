'''
ImprovedTrainer and train_with_improvements
This module provides an advanced training and validation pipeline for deep metric learning models, 
with a focus on triplet loss and comprehensive evaluation for tasks such as face or person re-identification.
Classes:
--------
ImprovedTrainer:
    A trainer class that encapsulates training and validation logic for models using triplet loss. 
    It includes advanced features such as L2 regularization, gradient clipping, adaptive thresholding, 
    detailed metric computation, and visualization tools.
    Methods:
    --------
    __init__(self, model, device, threshold=0.7):
        Initializes the trainer with the given model, device, and similarity threshold.
    train_one_epoch(self, train_loader, optimizer, criterion, epoch):
        Trains the model for one epoch using triplet loss, L2 regularization, and gradient clipping.
        Returns training loss statistics.
    validate_comprehensive(self, distorted_loader, reference_loader, epoch):
        Performs comprehensive validation:
            - Builds a reference gallery with multiple embeddings per identity.
            - Evaluates queries against the gallery using cosine similarity.
            - Computes accuracy, precision, recall, F1, ROC AUC, and optimal threshold.
            - Generates detailed plots (similarity distributions, ROC, PR, F1 vs threshold, confusion matrix, summary).
            - Visualizes embeddings with t-SNE for both true and predicted labels.
            - Analyzes and logs high-confidence failure cases.
        Returns a dictionary of validation metrics.
    _find_optimal_threshold(self, y_true, y_scores):
        Finds the similarity threshold that maximizes F1 score.
    _create_detailed_plots(self, similarities, ground_truth, predictions, fpr, tpr, roc_auc, epoch, optimal_threshold):
        Generates and saves comprehensive analysis plots for validation.
    _create_tsne_visualization(self, embeddings, true_labels, predicted_labels, epoch):
        Generates and saves t-SNE visualizations of embeddings colored by true and predicted labels.
    _analyze_failure_cases(self, similarities, ground_truth, predicted_labels, true_labels, epoch):
        Identifies and prints high-confidence failure cases where the model made incorrect predictions.
Function:
---------
train_with_improvements(model, train_loader, val_distorted_loader, val_reference_loader, device, epochs=25, lr=1e-4):
    Trains the given model using the ImprovedTrainer with advanced triplet loss, AdamW optimizer, 
    cosine annealing learning rate scheduler, and early stopping based on F1 score.
    Saves the best model checkpoint and prints progress.
    Returns the trained model.
Usage:
------
- Instantiate ImprovedTrainer with your model and device.
- Use train_with_improvements to run the full training and validation loop.
- Visualizations and analysis are automatically saved during validation.
Dependencies:
-------------
- torch
- numpy
- sklearn
- matplotlib
- seaborn
'''


import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

class ImprovedTrainer:
    def __init__(self, model, device, threshold=0.7, enable_plots=True):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.best_f1 = 0.0
        self.enable_plots = enable_plots
        
    def train_one_epoch(self, train_loader, optimizer, criterion, epoch):
        self.model.train()
        total_loss = 0
        triplet_losses = []
        
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            anchor = anchor.to(self.device)
            positive = positive.to(self.device) 
            negative = negative.to(self.device)
            
            optimizer.zero_grad()
            
            # Get embeddings
            anc_emb = self.model(anchor)
            pos_emb = self.model(positive)
            neg_emb = self.model(negative)
            
            # Calculate loss
            loss = criterion(anc_emb, pos_emb, neg_emb)
            
            # Add L2 regularization to prevent overfitting
            l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in self.model.parameters())
            total_loss_batch = loss + l2_reg
            
            total_loss_batch.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            triplet_losses.append(loss.item())
            
            # if batch_idx % 50 == 0:
            #     print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        return {
            'train_loss': avg_loss,
            'triplet_loss': np.mean(triplet_losses)
        }
    
    @torch.no_grad()
    def validate_comprehensive(self, distorted_loader, reference_loader, epoch):
        self.model.eval()
        
        # Step 1: Build reference gallery with multiple embeddings per identity
        ref_embeddings = defaultdict(list)
        ref_labels_map = {}
        
        print("Building reference gallery...")
        for imgs, labels in reference_loader:
            imgs = imgs.to(self.device)
            embeddings = self.model(imgs)
            
            for emb, label in zip(embeddings, labels):
                person_id = reference_loader.dataset.label_map[label.item()]
                ref_embeddings[person_id].append(emb.cpu())
                ref_labels_map[person_id] = label.item()
        
        # Use multiple embeddings per identity instead of just mean
        ref_gallery = {}
        for person_id, emb_list in ref_embeddings.items():
            ref_gallery[person_id] = torch.stack(emb_list)
        
        # Step 2: Evaluate on distorted images
        all_similarities = []
        all_predictions = []
        all_ground_truth = []
        all_predicted_identities = []
        query_embeddings = []
        true_identities = []
        
        print("Evaluating distorted queries...")
        for imgs, labels in distorted_loader:
            imgs = imgs.to(self.device)
            query_embs = self.model(imgs).cpu()
            query_embeddings.append(query_embs)
            
            for i, label in enumerate(labels):
                query_emb = query_embs[i]
                true_person = distorted_loader.dataset.label_map[label.item()]
                true_identities.append(true_person)
                
                # Find best match using maximum similarity across all reference images
                best_similarity = -1
                best_match_person = None
                
                for ref_person, ref_emb_stack in ref_gallery.items():
                    # Calculate similarity with all reference images for this person
                    similarities = F.cosine_similarity(
                        query_emb.unsqueeze(0).expand(ref_emb_stack.size(0), -1),
                        ref_emb_stack,
                        dim=1
                    )
                    max_sim = similarities.max().item()
                    
                    if max_sim > best_similarity:
                        best_similarity = max_sim
                        best_match_person = ref_person
                
                all_similarities.append(best_similarity)
                all_predicted_identities.append(best_match_person)
                
                # Ground truth: 1 if correctly matched, 0 otherwise
                is_correct = (best_match_person == true_person)
                all_ground_truth.append(1 if is_correct else 0)
                
                # Prediction: 1 if similarity > threshold AND correct match
                prediction = 1 if (best_similarity >= self.threshold and is_correct) else 0
                all_predictions.append(prediction)
        
        # Calculate metrics
        accuracy = accuracy_score(all_ground_truth, all_predictions)
        precision = precision_score(all_ground_truth, all_predictions, zero_division=0)
        recall = recall_score(all_ground_truth, all_predictions, zero_division=0)
        f1 = f1_score(all_ground_truth, all_predictions, zero_division=0)
        
        # Calculate ROC AUC
        fpr, tpr, _ = roc_curve(all_ground_truth, all_similarities)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold
        optimal_threshold = self._find_optimal_threshold(all_ground_truth, all_similarities)
        
        # Detailed analysis
        if self.enable_plots:
            self._create_detailed_plots(
                all_similarities, all_ground_truth, all_predictions,
                fpr, tpr, roc_auc, epoch, optimal_threshold
            )
            
            # t-SNE visualization with better labeling
            self._create_tsne_visualization(
                torch.cat(query_embeddings), true_identities, 
                all_predicted_identities, epoch
            )
            
            # Analysis of failure cases
            self._analyze_failure_cases(
                all_similarities, all_ground_truth, all_predicted_identities,
                true_identities, epoch
            )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'mean_positive_similarity': np.mean([s for s, gt in zip(all_similarities, all_ground_truth) if gt == 1]),
            'mean_negative_similarity': np.mean([s for s, gt in zip(all_similarities, all_ground_truth) if gt == 0]),
            'val_loss': 0.0  # Placeholder
        }
        
        print(f"Validation Results - Epoch {epoch}:")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        # print(f"ROC AUC: {roc_auc:.4f}, Optimal Threshold: {optimal_threshold:.4f}")
        
        return metrics
    
    def _find_optimal_threshold(self, y_true, y_scores):
        """Find threshold that maximizes F1 score"""
        thresholds = np.linspace(0, 1, 200)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            predictions = [1 if score >= threshold else 0 for score in y_scores]
            f1 = f1_score(y_true, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def _create_detailed_plots(self, similarities, ground_truth, predictions, 
                             fpr, tpr, roc_auc, epoch, optimal_threshold):
        """Create comprehensive visualization plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Similarity distribution
        pos_sims = [s for s, gt in zip(similarities, ground_truth) if gt == 1]
        neg_sims = [s for s, gt in zip(similarities, ground_truth) if gt == 0]
        
        axes[0, 0].hist(pos_sims, bins=50, alpha=0.7, label='Positive pairs', color='green')
        axes[0, 0].hist(neg_sims, bins=50, alpha=0.7, label='Negative pairs', color='red')
        axes[0, 0].axvline(self.threshold, color='black', linestyle='--', label=f'Threshold: {self.threshold}')
        axes[0, 0].axvline(optimal_threshold, color='blue', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Similarity Distribution')
        axes[0, 0].legend()
        
        # 2. ROC Curve
        axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(ground_truth, similarities)
        axes[0, 2].plot(recall, precision)
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].grid(True)
        
        # 4. F1 vs Threshold
        thresholds = np.linspace(0, 1, 200)
        f1_scores = []
        for t in thresholds:
            preds = [1 if s >= t else 0 for s in similarities]
            f1_scores.append(f1_score(ground_truth, preds, zero_division=0))
        
        axes[1, 0].plot(thresholds, f1_scores)
        axes[1, 0].axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score vs Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 5. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(ground_truth, predictions)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues')
        axes[1, 1].set_title('Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        # 6. Performance metrics summary
        metrics_text = f"""
        Accuracy: {accuracy_score(ground_truth, predictions):.4f}
        Precision: {precision_score(ground_truth, predictions, zero_division=0):.4f}
        Recall: {recall_score(ground_truth, predictions, zero_division=0):.4f}
        F1 Score: {f1_score(ground_truth, predictions, zero_division=0):.4f}
        ROC AUC: {roc_auc:.4f}
        
        Positive pairs: {len(pos_sims)}
        Negative pairs: {len(neg_sims)}
        Mean pos similarity: {np.mean(pos_sims):.4f}
        Mean neg similarity: {np.mean(neg_sims):.4f}
        """
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Metrics Summary')
        
        plt.tight_layout()
        plt.savefig(f'comprehensive_analysis_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_tsne_visualization(self, embeddings, true_labels, predicted_labels, epoch):
        """Create t-SNE visualization with better clustering analysis"""
        if len(embeddings) > 1000:  # Sample for computational efficiency
            indices = np.random.choice(len(embeddings), 1000, replace=False)
            embeddings = embeddings[indices]
            true_labels = [true_labels[i] for i in indices]
            predicted_labels = [predicted_labels[i] for i in indices]
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings.numpy())
        
        # Create subplots for true vs predicted labels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # True labels
        unique_true = list(set(true_labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_true)))
        for i, label in enumerate(unique_true):
            mask = np.array(true_labels) == label
            ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=label, alpha=0.7, s=20)
        ax1.set_title('t-SNE: True Labels')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Predicted labels
        unique_pred = list(set(predicted_labels))
        colors_pred = plt.cm.tab20(np.linspace(0, 1, len(unique_pred)))
        for i, label in enumerate(unique_pred):
            mask = np.array(predicted_labels) == label
            ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors_pred[i]], label=label, alpha=0.7, s=20)
        ax2.set_title('t-SNE: Predicted Labels')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'tsne_comparison_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # def _analyze_failure_cases(self, similarities, ground_truth, predicted_labels, 
    #                          true_labels, epoch):
    #     """Analyze and log failure cases"""
    #     failures = []
    #     for i, (sim, gt, pred_label, true_label) in enumerate(zip(similarities, ground_truth, predicted_labels, true_labels)):
    #         if gt == 0:  # Incorrect match
    #             failures.append({
    #                 'similarity': sim,
    #                 'predicted_identity': pred_label,
    #                 'true_identity': true_label,
    #                 'type': 'wrong_identity'
    #             })
        
    #     # Sort by similarity to find high-confidence wrong predictions
    #     high_conf_failures = sorted([f for f in failures if f['similarity'] > self.threshold], 
    #                               key=lambda x: x['similarity'], reverse=True)
        
    #     print(f"\nTop 5 high-confidence failures (Epoch {epoch}):")
    #     for i, failure in enumerate(high_conf_failures[:5]):
    #         print(f"{i+1}. Similarity: {failure['similarity']:.4f}, "
    #               f"Predicted: {failure['predicted_identity']}, "
    #               f"True: {failure['true_identity']}")

# Example usage in your training loop:
def train_with_improvements(model, train_loader, val_distorted_loader, val_reference_loader, 
                          device, epochs=25, lr=1e-4):
    
    # Use improved loss function
    from B_model import ImprovedTripletLoss
    criterion = ImprovedTripletLoss(margin=0.3, hard_mining=True)
    
    # Use adaptive learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    trainer = ImprovedTrainer(model, device, threshold=0.7, enable_plots=False)
    
    best_f1 = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        train_metrics = trainer.train_one_epoch(train_loader, optimizer, criterion, epoch)
        
        # Validation
        val_metrics = trainer.validate_comprehensive(val_distorted_loader, val_reference_loader, epoch)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"[Epoch {epoch}] Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, "
            f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")

    
    return model