# Facial Verification System for Vehicle Protection


## Description

This project aims to develop an integrated facial verification system
for securing access to a robot-car (demonstration vehicle). The system
combines artificial intelligence with hardware components to enable
real-time user authentication.
<p align="center">
  <img src="https://github.com/user-attachments/assets/b6a2543c-ed51-4753-a279-46562a862274" alt="img1" width="300"/>
  <img src="https://github.com/user-attachments/assets/bbe85ec6-300c-42e2-8438-8a9d088fddd3" alt="img2" width="300"/>
  <img src="https://github.com/user-attachments/assets/d5c87de3-96bd-4bdd-ade4-d655906aa348" alt="img3" width="300"/>
</p>



## Requirements

-   Identification of pre-authorized persons through automatic facial
    verification.\
-   Control of a robot-car based on user recognition.\
-   Vehicle access security by integrating AI and hardware.

## Technical Solution

-   Siamese neural network for facial verification.\
-   STM32 microcontroller for robot hardware control.\
-   Graphical User Interface (GUI) application:
    -   Management of authorized users.\
    -   Bluetooth communication with the robot.\
-   Automatic fine-tuning of the model for retraining when a new user is
    added.

## Results

-   Functional facial verification system with an accuracy of 84.2%.\
-   Fully operational robot-car equipped with:
    -   obstacle avoidance sensors,\
    -   Bluetooth control,\
    -   user management application.

## Testing and Validation

-   Model evaluation using accuracy, precision, recall, and F1-score.\
-   Authentication tests under different lighting conditions.\
-   Verification of Bluetooth communication and hardware control.

## Personal Contributions

-   Implementation of the Siamese neural network for facial
    verification.\
-   Development of the GUI application.\
-   Programming of the STM32 microcontroller.\
-   Assembly of the robot and integration of hardware components.\
-   Establishing Bluetooth communication between the application and the
    robot.

## References

-   Koch et al. (2015) -- *Siamese Neural Networks for One-Shot Image
    Recognition*\
    [Link PDF](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)\
-   STMicroelectronics documentation -- STM32F303RE\
-   Frameworks: TensorFlow / Keras

------------------------------------------------------------------------

Project developed at the Faculty of Automation and Computers --
Department of Computer Science.
