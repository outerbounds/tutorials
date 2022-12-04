from metaflow import FlowSpec, Parameter, step, batch, environment, S3, metaflow_config, current

class TrainHandGestureClassifier(FlowSpec):

    S3_URI = Parameter(
        's3', type=str, 
        default='s3://outerbounds-tutorials/computer-vision/hand-gesture-recognition',
        help = 'The s3 uri to the root of the model objects.'
    )

    DATA_ROOT = Parameter(
        'data', type=str, default='data/',
        help = 'The relative location of the training data.'
    )

    IMAGES = Parameter(
        'images', type=str,
        default = 'subsample.zip',
        help = 'The path to the images.'
    )

    ANNOTATIONS = Parameter(
        'annotations', type=str,
        default = 'subsample-annotations.zip'
    )

    PATH_TO_CONFIG = Parameter(
        'config', type=str, 
        default = 'hagrid/classifier/config/default.yaml',
        help = 'The path to classifier training config.'
    )
    
    NUMBER_OF_EPOCHS = Parameter(
        'epochs', type=int, default=100,
        help = 'The number of epochs to train the model from.'
    )

    MODEL_NAME = Parameter(
        'model', type=str,
        default = 'MobileNetV3_small',
        help = '''Pick a model from:
            - [ResNet18, ResNext50, ResNet152, MobileNetV3_small, MobileNetV3_large, Vitb32]
        '''
    )

    CHECKPOINT_PATH = Parameter(
        'checkpoint', type=str, default = None, 
        help = 'Path to the model state you want to resume. Eithe'
    )

    @step
    def start(self):
        # Configure the (remote) experiment tracking location.
        # In this tutorial, experiment tracking means
            # 1: Storing the best model state checkpoints to S3.
            # 2: Storing parameters as Metaflow artifacts.
            # 3: Storing metrics/logs with Tensorboard. 
        import os
        print("Training {} in flow {}".format(self.MODEL_NAME, current.flow_name))
        self.datastore = metaflow_config.METAFLOW_CONFIG['METAFLOW_DATASTORE_SYSROOT_S3']
        self.experiment_storage_prefix = os.path.join(self.datastore, current.flow_name, current.run_id)
        self.next(self.train)

    def _download_data_from_s3(self, file, sample : bool = True):
        import zipfile
        import os
        with S3(s3root = self.S3_URI) as s3:
            if sample:
                path = os.path.join(self.DATA_ROOT, file)
                result = s3.get(path)
                with zipfile.ZipFile(result.path, 'r') as zip_ref:
                    zip_ref.extractall(path.split('.zip')[0])
            else: # Full dataset is not yet available.
                raise NotImplementedError()

    @batch(
        gpu=1,
        memory=32000,
        image='eddieob/cv-tutorial:gpu-latest',
        shared_memory=8000,
    )
    @step
    def train(self):

        from hagrid.classifier.run import run_train
        from hagrid.classifier.utils import get_device
        import os
        
        # Download the dataset onto the compute instance.
        if not os.path.exists(self.DATA_ROOT):
            os.mkdir(self.DATA_ROOT)
        print("Downloading images...")
        self._download_data_from_s3(self.IMAGES, sample=True)
        print("Done!")
        print("Downloading annotations...")
        self._download_data_from_s3(self.ANNOTATIONS, sample=True)
        print("Done!")

        # Train a model from available MODEL_NAME options from a checkpoint.
        # Pytorch errors will happen if CHECKPOINT_PATH doesn't match MODEL_NAME.
        self.train_args = dict(
            path_to_config = self.PATH_TO_CONFIG,
            number_of_epochs = self.NUMBER_OF_EPOCHS,
            device = get_device(),
            checkpoint_path = self.CHECKPOINT_PATH,
            model_name = self.MODEL_NAME,
            tensorboard_s3_prefix = self.experiment_storage_prefix,
            always_upload_best_model = True
        )
        _ = run_train(**self.train_args)

        # Move the best model checkpoint to S3. 
        # See the comment in the start step about setting self.experiment_storage_prefix.
        experiment_path = os.path.join("experiments", self.MODEL_NAME)
        path_to_best_model = os.path.join(experiment_path, 'best_model.pth')
        self.best_model_location = os.path.join(self.experiment_storage_prefix, path_to_best_model)
        if self.best_model_location.startswith('s3://'):
            with S3(s3root = self.experiment_storage_prefix) as s3:
                s3.put_files([(path_to_best_model, path_to_best_model)])
                print("Best model checkpoint saved at {}".format(self.best_model_location))
        self.next(self.end)
        
    @step
    def end(self):
        pass

if __name__ == '__main__':
    TrainHandGestureClassifier()
