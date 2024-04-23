def process_image(image):
    # Preprocess the image
    resized_image = image.resize((300, 300))
    normalized_image = np.array(resized_image) / 255.0
    processed_image = np.expand_dims(normalized_image, axis=0)

    prediction = best_model.predict(processed_image)
    print(prediction)
    result_label = "Fake" if prediction < 0.5 else "Real"
    return result_label