# Visualizing a Point Cloud Using Scene Depth

Place points in the real-world using the scene's depth data to visualize the shape of the physical environment.  

## Overview

- Note: This sample code project is associated with WWDC20 session [10611: Explore ARKit 4](https://developer.apple.com/wwdc20/10611/).

## Configure the Sample Code Project

Before you run the sample code project in Xcode, set the run destination to an iPad Pro with a LiDAR sensor, running iPadOS 14.0 or later.

## To extract RGB, Depth and Point Cloud

pixelBuffer, depthMap : Defines RGB and Depth information.

Point Cloud data is created by applying the projection function (matrix specified by sensors) written in the code.

Each 2D pixel, RGB, depth, and 3D coordinates are stored as an array.

![image](https://user-images.githubusercontent.com/35245580/112245845-09f50380-8c95-11eb-9dda-b79e717379d3.png)
