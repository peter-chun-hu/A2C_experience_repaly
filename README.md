# **A2C_experience_repaly**
- - -
## **Introduction**
* Implemented A2C algorithm with experience replay
* A2C algorithm WITHOUT experience replay is the baseline in our results
## **Results**
* MountainCar-v0  
     * Different replay buffer size
     ![Alt text](imgs/Mountaincar_buff.jpg)  
     * Different sample number  
     ![Alt text](imgs/Mountaincar_sample_size.jpg)  
     * Prioritized or not  
     ![Alt text](imgs/Mountaincar_prioritized.jpg)  
* CartPole-v1  
     * Different replay buffer size  
     ![Alt text](imgs/carpole_buffer.jpg)  
     * Different sample number  
     ![Alt text](imgs/carpole_sample.jpg)  
     * Prioritized or not  
     ![Alt text](imgs/carpole_prioritize.jpg)  
* Video (Click the image below to play on YouTube)  
     [![](http://img.youtube.com/vi/mIvstl3QufM/0.jpg)](http://www.youtube.com/watch?v=mIvstl3QufM)
* [Report](ece-276-final.pdf)

## **Environment**
* ### Install and run docker with ```sudo docker run -it fraserlai/276_project:gym_10_TA_v6 /bin/bash```

## **Requirements**
* Python 3
* PyTorch
* OpenAI baselines
* Anaconda

## **Run** ##
* Open and run *main_experience_replay.ipynb*

## **Reference** ##
* [openai baseline](https://github.com/openai/baselines/tree/master/baselines/a2c)
* [ikostrikov pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)
