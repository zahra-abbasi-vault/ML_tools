import os
from tensorflow import keras
from model_utils import my_models
from tensorflow import keras
import tensorflow as tf
import absl.logging
import os


LOCAL_MODELS_DIR = r""

K = keras.backend
layers = keras.layers
monitor    = 'val_accuracy' 

class SaveBestAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, file_path):
        super(SaveBestAccuracyCallback, self).__init__()
        self.file_path = file_path
        self.best_accuracy = 0.0
        self.best_loss = 100

    def on_epoch_end(self, epoch, logs=None):
        if monitor == 'val_accuracy':
            current_accuracy = logs.get('val_accuracy')
            if current_accuracy is not None and current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                with open(self.file_path, 'w') as file:
                    file.write(f'\n\nEpoch {epoch + 1}: '
                            f'Training Acc - {logs["accuracy"]:.4f}, '
                            f'Validation Acc - {logs["val_accuracy"]:.4f}, '
                            f'Training Loss - {logs["loss"]:.4f}, '
                            f'Validation Loss - {logs["val_loss"]:.4f}\n')
        if monitor == 'val_loss':
            current_loss = logs.get('val_loss')
            if current_loss is not None and current_loss < self.best_loss:
                self.best_loss = current_loss
                with open(self.file_path, 'w') as file:
                    file.write(f'\n\nEpoch {epoch + 1}: '
                            f'Training Acc - {logs["accuracy"]:.4f}, '
                            f'Validation Acc - {logs["val_accuracy"]:.4f}, '
                            f'Training Loss - {logs["loss"]:.4f}, '
                            f'Validation Loss - {logs["val_loss"]:.4f}\n')



def train_main(
    output_shape ,
    train_id ,
    input_shape ,
    checkpoint_id ,
    train_generator,
    val_generator,
    steps_per_epoch
):
    absl.logging.set_verbosity(absl.logging.ERROR)
    physical_devices = tf.config.list_physical_devices('GPU')
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

    EPOCHS     = 5000
    BATCH_SIZE = 32


    MODEL_BUILDER = my_models.Model_1
    model = MODEL_BUILDER(input_shape, output_shape)
    model.summary()


    model_name  = f"{MODEL_BUILDER.__name__}-{train_id}"
    chkpnt_name = os.path.join(LOCAL_MODELS_DIR, "weights" , model_name , checkpoint_id)

    checkpoint  = keras.callbacks.ModelCheckpoint(chkpnt_name, monitor=monitor, verbose=1, save_best_only=True, save_weights_only=False, save_freq='epoch', period=1)
    tensorboard = keras.callbacks.TensorBoard(os.path.join(LOCAL_MODELS_DIR, "tensorboard", model_name , checkpoint_id))
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=10, min_lr=1e-7 , verbose=1)
    save_log = SaveBestAccuracyCallback(chkpnt_name+"accuracy.txt")
    if not os.path.exists(chkpnt_name): os.makedirs(chkpnt_name)     


        
    model.compile(
        optimizer=
        keras.optimizers.SGD(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'], run_eagerly=False)

    model.fit(
        train_generator.generate_batches(BATCH_SIZE, augment=True),
        validation_data  = val_generator.get_all_data(),
        epochs           = EPOCHS,
        steps_per_epoch  = steps_per_epoch,
        callbacks        = [checkpoint, tensorboard, reduce_lr , save_log ])
