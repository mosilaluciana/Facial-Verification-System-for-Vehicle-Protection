import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils.preprocessing import preprocess_twin_train, preprocess_twin_val

def binary_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

@tf.function
def train_step(batch, model, optimizer):
    with tf.GradientTape() as tape:
        x = batch[:2]
        y = batch[2]
        y_pred = model(x, training=True)
        loss = binary_loss(y, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, y_pred

def train_model(model, train_data, val_data, epochs=50):


    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    train_auc_history = []
    val_auc_history = []

    checkpoint_dir = './training_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        print(f'\nEpoca {epoch + 1}/{epochs}')
        train_losses, train_accs = [], []
        train_labels_all, train_preds_all = [], []

        for i, batch in enumerate(train_data):
            loss, y_pred = train_step(batch, model, optimizer)
            loss_value = tf.reduce_mean(loss).numpy()
            train_losses.append(loss_value)

            y_true = batch[2].numpy()
            y_pred_np = y_pred.numpy()
            y_pred_labels = (y_pred_np > 0.5).astype(np.float32)
            acc = np.mean(y_true == y_pred_labels)
            train_accs.append(acc)
            train_labels_all.append(y_true)
            train_preds_all.append(y_pred_np)

        train_labels_all = np.concatenate(train_labels_all)
        train_preds_all = np.concatenate(train_preds_all)
        try:
            train_auc = roc_auc_score(train_labels_all, train_preds_all)
        except:
            train_auc = 0.0

        val_losses, val_accs = [], []
        val_labels_all, val_preds_all = [], []

        for batch in val_data:
            x = batch[:2]
            y_true = batch[2].numpy()
            y_pred = model(x, training=False)
            loss = binary_loss(y_true, y_pred).numpy()
            val_losses.append(np.mean(loss))

            y_pred_np = y_pred.numpy()
            y_pred_labels = (y_pred_np > 0.5).astype(np.float32)
            acc = np.mean(y_true == y_pred_labels)
            val_accs.append(acc)
            val_labels_all.append(y_true)
            val_preds_all.append(y_pred_np)

        val_labels_all = np.concatenate(val_labels_all)
        val_preds_all = np.concatenate(val_preds_all)
        try:
            val_auc = roc_auc_score(val_labels_all, val_preds_all)
        except:
            val_auc = 0.0

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_train_acc = np.mean(train_accs)
        avg_val_acc = np.mean(val_accs)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_acc_history.append(avg_train_acc)
        val_acc_history.append(avg_val_acc)
        train_auc_history.append(train_auc)
        val_auc_history.append(val_auc)

        current_lr = float(tf.keras.backend.get_value(optimizer.learning_rate))
        print(f'Rezumat epoca {epoch + 1}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, ROC-AUC: {train_auc:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, ROC-AUC: {val_auc:.4f}')
        print(f'  Learning Rate: {current_lr:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save_weights('./best_model_weights.h5')
            print('  ✓ Cel mai bun modelvechi salvat!')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping la epoca {epoch + 1}')
            break

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=os.path.join(checkpoint_dir, 'ckpt'))
            print(f'  Checkpoint salvat la epoca {epoch + 1}')

    try:
        model.load_weights('./best_model_weights.h5')
        print('Cel mai bun model vechi încarcat pentru utilizare finala.')
    except:
        print('Nu s-a putut încarca cel mai bun modelvechi, se folosește ultimul.')

    return {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'train_acc': train_acc_history,
        'val_acc': val_acc_history,
        'train_auc': train_auc_history,
        'val_auc': val_auc_history
    }

def plot_training_history(history):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history['train_acc'], label='Train Accuracy')
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(history['train_auc'], label='Train ROC-AUC')
        axes[2].plot(history['val_auc'], label='Validation ROC-AUC')
        axes[2].set_title('Model ROC-AUC')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("Graficul a fost salvat ca 'training_history.png'")
        plt.close()
    except Exception as e:
        print(f"[Eroare] Nu s-a putut crea graficul: {e}")
