from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# microsoft/trocr-base-printed
# microsoft/trocr-base-handwritten

pretrain_model = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(pretrain_model)
model = VisionEncoderDecoderModel.from_pretrained(pretrain_model)


def read_text_from_image_text(image_text, processor, model):
    pixel_values = processor(image_text, return_tensors="pt",attention_mask=1).pixel_values
    #
    generated_ids = model.generate(pixel_values)
    #
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def read_all_text_from_images_text(images_text):
    # pretrain_model = "microsoft/trocr-base-printed"
    # if processor is None: processor = TrOCRProcessor.from_pretrained(pretrain_model)
    # if model is None: model = VisionEncoderDecoderModel.from_pretrained(pretrain_model)

    return [read_text_from_image_text(image_text, processor, model) for image_text in images_text]


