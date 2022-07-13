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
   <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The goal of this project is to to turn people into a game controller by playing games using body poses.
You can choose any pose you like, map it to any keyboard or mouse input and play your favourite games with your desired poses at no extra cost.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

In order to use Human Controller you will need to have:
* Working video camera
* Video game (choose any game and the app simulates the keys)

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
2. Train a model to predict your poses (takes about 0.5x+2 minutes where x is the number of poses).
3. Press play to start playing and let the app run in the background while your game runs.

#### GUI example:
<p align="center">
  <img width="670" alt="gui" src="https://user-images.githubusercontent.com/80704907/171219083-595c3876-108f-4413-ac08-6dcbfca51d3c.png"/>
</p>

<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgments

We wish to thank our mentor Omri Avrahami for his guidance and support at any time.

<p align="right">(<a href="#top">back to top</a>)</p>
