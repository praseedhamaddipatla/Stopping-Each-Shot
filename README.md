# Stopping-Each-Shot
A machine learning-based gun detection system utilizing neural networks to identify open and concealed firearms in standard and infrared video frames with 98.5% accuracy. 
Features an infrared imaging approach for concealed weapon identification and a mobile app designed for real-time shooter alerts and low-cost community safety.

**Abstract:**
Gun violence is a national epidemic in the United States, with recent years typically
seeing over 44,000 casualties, a number that is only increasing. For all involved, these
shootings have an unfathomable negative impact, causing mass fear, trauma, and
lasting mental health issues. Current preventative measures are expensive, requiring
excessive human surveillance and producing inadequate reaction time. To provide a
low-cost and productive response, a machine learning system was developed using a
neural network to identify various gun models in standard and infrared video frames.
Infrared imaging detects differences in thermal radiation wavelengths, exposing
concealed weapons on the human body. To effectively detect both open and concealed
guns, an infrared camera was utilized to create a dataset of 1,000 photos of a concealed
gun, with over 40,000 additional images of unconcealed guns gathered from the
internet. All of these photos were from various angles to imitate real-life scenarios.
Once data preparation was completed, all images were applied for training, creating a
set of weights for the neural network, leading to successful recognition within the
testing phase. Statistical analyses indicated a 0.985 accuracy and a precision value of
0.977, demonstrating exemplary performance. For wide-scale implementation, a mobile
application of the model was deployed that allows users to determine whether a gun is
present, receive notifications of potential shooters, and alert authorities as necessary.
This study would help increase the efficiency of emergency responses and provide a
low-cost, accessible option for underprivileged communities.
