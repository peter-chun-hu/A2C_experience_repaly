# **A2C_experience_repaly**
- - -
## **Introduction**
* Implemented A2C algorithm with experience replay
* A2C algorithm WITHOUT experience replay is the baseline in our results
## **Results**
* ### MountainCar-v0  
     * Different replay buffer size  
       <img src="imgs/Mountaincar_buff.jpg" width="575">  
     * Different sample number  
       <img src="imgs/Mountaincar_sample_size.jpg" width="575">  
     * Prioritized or not  
       <img src="imgs/Mountaincar_prioritized.jpg" width="575">  
* ### CartPole-v1  
     * Different replay buffer size  
       <img src="imgs/carpole_buffer.jpg" width="575">  
     * Different sample number  
       <img src="imgs/carpole_sample.jpg" width="575">  
     * Prioritized or not  
       <img src="imgs/carpole_prioritize.jpg" width="575">  
* Video (Click the image below to play on YouTube)  
     [![](http://img.youtube.com/vi/mIvstl3QufM/0.jpg)](http://www.youtube.com/watch?v=mIvstl3QufM)
* [Report](ece-276-final.pdf)

## **Environment**
* Install and run docker with ```sudo docker run -it fraserlai/276_project:gym_10_TA_v6 /bin/bash```

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
