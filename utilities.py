import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image, ImageFilter

def text_encoding(input_description, excel_file, top_n=5):
    df = pd.read_excel(excel_file)
    if 'Line Content' not in df.columns or 'File Name' not in df.columns:
        raise ValueError("The Excel file must contain 'Line Content' and 'File Name' columns.")

    all_descriptions = df['Line Content'].tolist()
    all_descriptions.append(input_description)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)
    
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    top_indices = similarity_scores.argsort()[-top_n:][::-1] 

    top_matches = []
    for index in top_indices:
        file_name = df.iloc[index]['File Name']
        line_content = df.iloc[index]['Line Content']
        score = similarity_scores[index]
        top_matches.append((file_name, line_content, score))

    print(f"Input Description: {input_description}")
    print("\nTop Matches:")
    for i, (file_name, line_content, score) in enumerate(top_matches, 1):
        print(f"{i}. File Name: {file_name}, Description: {line_content}, Similarity Score: {score:.4f}")
    
    return [file_name for file_name, _, _ in top_matches]



from PIL import Image, ImageFilter
import os

def save_images(image_paths,  blur_radius=2, image_size=(256, 256)):
    save_folder="C:\\Users\\Public\\Documents\\Text-Image-GAN\\app\\static\\generated_images"
    os.makedirs(save_folder, exist_ok=True)
    
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        img_resized = img.resize(image_size)
        blurred_img = img_resized.filter(ImageFilter.GaussianBlur(blur_radius))
        
        save_path = os.path.join(save_folder, f"generated_image_{i+1}.png")
        blurred_img.save(save_path)
        print(f"Saved blurred image: {save_path}")



