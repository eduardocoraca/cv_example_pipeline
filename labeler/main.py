from image_labeler import ImageLabeler

image_labeler = ImageLabeler(
    src_path=r"data\unlabeled", dst_path=r"data\labels", out_path=r"data\images"
)

image_labeler.label_dir()
