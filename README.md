<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">

<h2 align="center">Human Controller</h2>

  <p align="center">
    Playing video games with body poses
    <br />
    <a href="https://youtu.be/IPQhQQrVtcg"><strong>Demo video</strong></a>
    ·
    <a href="https://youtu.be/IGle3bZDcw8"><strong>Extended video</strong></a>
    <br />
  </p>
</div>

<p align="center">
  <img src="https://user-images.githubusercontent.com/80704907/154738336-7d4aa904-3f71-43cc-b7b7-f2b467988a45.gif" alt="animated" />
</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
   </li>
   <li>
     <a href="#getting-started">Getting Started</a>
     <ul>
     <li><a href="#prerequisites">Prerequisites</a></li>
      <li><a href="#installation">Installation</a></li>
     </ul>
   </li>
   <li><a href="#usage">Usage</a></li>
   <li><a href="#future-work">Future Work</a></li>
   <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The goal of this project to create a virtual game controller for playing video games with body poses. 
You can choose any pose you like, map it to any keyboard or mouse input and play your favourite games with your desired poses at no extra cost.
For the full project report check out the project_book.pdf.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

In order to use Human Controller you will need to have:
* Working video camera
* Video game (choose any game and the app simulates the keys)
* Since some games require DirectInput keypresses, this project is compatible with the Windows OS.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/YoavOrenbach/HumanController.git
   ```
2. Install reqired packages
   ```sh
   pip install requirements.txt
   ```
3. Run main.py to use the app
   ```sh
   python main.py
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

There are 3 steps in order to use Human Controller:
1. Take poses you would like to use in game and map them to any keyboard or mouse input.
2. Train a model to predict your poses (takes about 0.33x+1 minutes where x is the number of poses).
3. Press play to start playing and let the app run in the background while your game runs.

#### GUI example:
<p align="center">
  <img width="670" alt="gui" src="https://user-images.githubusercontent.com/80704907/171219083-595c3876-108f-4413-ac08-6dcbfca51d3c.png"/>
</p>

#### Pipeline:
<p align="center">
  <img width="670" alt="pipeline" src="https://user-images.githubusercontent.com/80704907/185976219-939b8e67-6c45-43f1-a0dd-16115c8b5a51.png"/>
</p>
After testing many methods we found that the optimal framework is passing the image of a pose to a real-time pose estimation model to extract keypoints on the human body, feature engineering processing these points, and passing them to a neural network for the final classification.

As default we use the models that achieved the best performance:
 - Pose estimation: MoveNet.
 - Feature Engineering: Normalizing keypoints + pariwise distances.
 - Classifier: stacking ensemble of multilayer perceptrons.
 
However, it is possible to run the program with any other model or method that we tested as detailed in the project book.

Use '-p' for pose estimation, '-f' for feature engineering , and '-c' for classifier, where all available options are in factory.py for instance:
```sh
python main.py -p blazepose -f normalization -c knn
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Future Work
There are three abstract classes for the three major components of our pipeline, and it is possible to extend them with new and improved pose estimation models, feature engineering methods, and classifiers for better pose classification.
In addition, multi-person pose estimation models could extend the class as well, allowing a two-player experience.

<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgments

We wish to thank our mentor Omri Avrahami for his guidance and support at any time.

<p align="right">(<a href="#top">back to top</a>)</p>
