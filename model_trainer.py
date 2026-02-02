from fastai.vision.all import *
import timm
import warnings
warnings.filterwarnings('ignore')

def main():
    path = Path(r"KrishiXpert\Plant Village")
    dls = ImageDataLoaders.from_folder(
        path,
        valid_pct=0.2,
        bs=16,
        item_tfms=Resize(460),
            batch_tfms=aug_transforms(size=224),
        num_workers=0
    )


    learn = vision_learner(dls, 'convnextv2_tiny', metrics=accuracy)
    learn.fine_tune(5, base_lr=3e-3)
    learn.export('plant_disease_classifier.pkl')
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(10, 10))
    plt.show()

if __name__ == '__main__':
    main()