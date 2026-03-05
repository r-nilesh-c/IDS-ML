# Learning Resources

## Books

### Machine Learning & Deep Learning
- **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron
  - Chapters 8-9: Dimensionality Reduction & Unsupervised Learning
  - Chapter 14: Deep Computer Vision (autoencoders)

- **"Deep Learning"** by Ian Goodfellow, Yoshua Bengio, Aaron Courville
  - Chapter 14: Autoencoders (comprehensive theory)

### Intrusion Detection
- **"Network Intrusion Detection and Prevention"** by Ali A. Ghorbani et al.
  - Covers IDS fundamentals and evaluation metrics

- **"Machine Learning and Security"** by Clarence Chio & David Freeman
  - Chapter 6: Anomaly Detection

## Online Courses

### Coursera
- **"Deep Learning Specialization"** by Andrew Ng
  - Course 1: Neural Networks and Deep Learning
  - Course 2: Improving Deep Neural Networks

- **"Machine Learning"** by Andrew Ng
  - Week 8: Unsupervised Learning (anomaly detection)

### Fast.ai
- **"Practical Deep Learning for Coders"**
  - Hands-on approach to building neural networks

## Papers

### Foundational Papers

1. **Isolation Forest**
   - Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008)
   - "Isolation Forest" - IEEE ICDM
   - [Link](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)

2. **Autoencoders for Anomaly Detection**
   - Sakurada, M., & Yairi, T. (2014)
   - "Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction"
   - MLSDA Workshop

3. **Deep Learning for IDS**
   - Vinayakumar, R., et al. (2019)
   - "Deep Learning Approach for Intelligent Intrusion Detection System"
   - IEEE Access

### CIC-IDS Datasets

4. **CIC-IDS2017**
   - Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018)
   - "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"
   - ICISSP 2018

5. **CIC-IDS2018**
   - Sharafaldin, I., Lashkari, A. H., Hakak, S., & Ghorbani, A. A. (2019)
   - "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy"
   - ICISSp 2019

### Healthcare Security

6. **Healthcare IDS**
   - Hady, A. A., et al. (2020)
   - "Intrusion Detection System for Healthcare Systems Using Medical and Network Data"
   - IEEE Access

## Datasets

### Network Intrusion Detection
- **CIC-IDS2017**: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- **CIC-IDS2018**: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html)
- **NSL-KDD**: Classic benchmark dataset
- **UNSW-NB15**: Modern network traffic dataset

### Anomaly Detection (General)
- **KDD Cup 1999**: Historical but still useful
- **CICIDS2012**: Earlier version of CIC datasets

## Tools & Libraries

### Python Libraries
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Visualization

### IDS-Specific Tools
- **Zeek (formerly Bro)**: Network analysis framework
- **Suricata**: IDS/IPS engine
- **Snort**: Network intrusion detection system

## Tutorials & Blogs

### Autoencoder Tutorials
- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
- [Anomaly Detection with Autoencoders](https://www.tensorflow.org/tutorials/generative/autoencoder)

### Isolation Forest
- [Isolation Forest in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Understanding Isolation Forest](https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e)

### IDS & Security
- [SANS Reading Room](https://www.sans.org/reading-room/) - Security papers
- [OWASP](https://owasp.org/) - Web application security

## Video Resources

### YouTube Channels
- **StatQuest with Josh Starmer**: Excellent ML explanations
  - Neural Networks series
  - Random Forests and Decision Trees

- **3Blue1Brown**: Visual mathematics
  - Neural Networks series (intuitive explanations)

- **Sentdex**: Python programming and ML
  - Deep Learning with TensorFlow

### Conference Talks
- **DEF CON**: Security conference talks
- **Black Hat**: Information security conferences
- **IEEE Symposium on Security and Privacy**

## Communities & Forums

### Online Communities
- **Stack Overflow**: Programming questions
- **Cross Validated**: Statistics and ML questions
- **Reddit**:
  - r/MachineLearning
  - r/netsec
  - r/deeplearning

### GitHub Repositories
- [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning)
- [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning)
- [Awesome Anomaly Detection](https://github.com/hoya012/awesome-anomaly-detection)

## Practice Platforms

### Kaggle
- Competitions and datasets
- Notebooks for learning
- Community discussions

### Google Colab
- Free GPU access
- Jupyter notebooks in the cloud
- Easy sharing and collaboration

## Documentation

### Official Documentation
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)

## Evaluation Metrics

### Understanding Metrics
- **Precision, Recall, F1-Score**: [scikit-learn guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- **ROC Curves and AUC**: [Understanding ROC curves](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- **Confusion Matrix**: [Visual guide](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)

## Healthcare-Specific Resources

### Standards & Regulations
- **HIPAA**: Health Insurance Portability and Accountability Act
- **HITECH**: Health Information Technology for Economic and Clinical Health Act
- **NIST Cybersecurity Framework**: Healthcare sector guidance

### Healthcare Security
- [HHS Cybersecurity Program](https://www.hhs.gov/about/agencies/asa/ocio/cybersecurity/index.html)
- [Healthcare Information and Management Systems Society (HIMSS)](https://www.himss.org/resources/cybersecurity)

## Recommended Learning Path

### Beginner (Weeks 1-2)
1. Python basics (if needed)
2. NumPy and pandas tutorials
3. Basic ML concepts (Coursera ML course, Week 1-3)
4. Start Lesson 1 of this module

### Intermediate (Weeks 3-4)
1. Neural networks fundamentals (3Blue1Brown series)
2. TensorFlow/Keras tutorials
3. Lessons 2-3 of this module
4. Practice on Kaggle datasets

### Advanced (Weeks 5-6)
1. Read foundational papers
2. Lessons 4 of this module
3. Implement variations and improvements
4. Contribute to open-source IDS projects

## Tips for Success

1. **Hands-on Practice**: Code along with tutorials
2. **Start Simple**: Master basics before advanced topics
3. **Read Papers**: Understand theoretical foundations
4. **Join Communities**: Ask questions and help others
5. **Build Projects**: Apply knowledge to real problems
6. **Stay Updated**: Follow ML/security news and blogs

## Additional Resources

### Cheat Sheets
- [NumPy Cheat Sheet](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [scikit-learn Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
- [Keras Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Keras_Cheat_Sheet_Python.pdf)

### Visualization Tools
- **TensorBoard**: TensorFlow visualization toolkit
- **Netron**: Neural network visualizer
- **Yellowbrick**: ML visualization library

## Keep Learning!

Machine learning and cybersecurity are rapidly evolving fields. Stay curious, keep experimenting, and don't be afraid to make mistakes - they're the best learning opportunities!

**Remember**: The goal isn't to memorize everything, but to understand core concepts and know where to find information when you need it.
