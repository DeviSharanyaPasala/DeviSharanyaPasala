# Devi Sharanya Pasala

I’m a graduate student in the MS Information Science program (Data Analytics & AI) at the University at Albany (SUNY).

Most of my projects come from coursework, research ideas, or problems I was curious about and wanted to understand better. I usually start with messy data, try a few modeling approaches, evaluate what actually works, and then document the results honestly. Some projects are more polished than others, but all of them reflect things I’ve implemented and tested myself.



## Projects I’ve Worked On

### Breast Cancer Detection using Deep Learning
This project was an attempt to understand how different convolutional neural network architectures behave on mammography data. I trained and compared multiple models and evaluated them using metrics that are commonly discussed in healthcare settings, such as precision, recall, AUC, and Cohen’s kappa. One of the main takeaways for me was how misleading accuracy alone can be when classes are imbalanced.

Repository: **Breast-Cancer-Detection**


### Patient Case Similarity Analysis
In this project, I worked on measuring similarity between patient case descriptions using basic NLP techniques. I used TF-IDF representations and cosine similarity to retrieve similar cases, and later explored clustering to see how cases grouped together. I also built a small Flask interface to make the idea easier to interact with, even though the core focus was on the similarity logic rather than the UI.

Repository: **Patient-Case-Similarity**


### Video Anomaly Detection
This project explores anomaly detection in video data using a ConvLSTM autoencoder. The model was trained on normal behavior and anomalies were identified using reconstruction error. I tested this on the UCSD pedestrian datasets and spent a fair amount of time tuning thresholds and understanding failure cases, especially false positives in crowded scenes.

Repository: **Spotting-Anomalous-Behaviour**


### Retinal Blood Vessel Segmentation
Here, I compared several deep learning architectures for retinal blood vessel segmentation, including U-Net variants, SegNet, DeepLab, and LR-ASPP. The goal was not to propose a new model, but to benchmark existing ones using Dice score, IoU, and AUC, and to understand trade-offs between segmentation quality and model complexity.

Repository: **Retinal-Vessel-Segmentation-Using-Deep-Learning**


### Product Review Scraper
This is a smaller, more practical project where I built a Flask-based tool to scrape product reviews from e-commerce pages. Since scraping can be unreliable, I added a demo mode to handle cases where requests are blocked. This project helped me better understand real-world data collection challenges.

Repository: **Product-Review-Scraper**


## Technical Background
- **Languages & Tools:** Python, SQL, Git  
- **Libraries:** Pandas, NumPy, scikit-learn, PyTorch, TensorFlow  
- **Other:** Flask, OpenCV  


## Current Interests
- Medical image analysis and model evaluation  
- NLP applications in healthcare  
- Behavioral data and health-related information nudges  


## Contact
- LinkedIn: https://www.linkedin.com/in/devi-sharanya/  
- GitHub: https://github.com/DeviSharanyaPasala
