# Visualizing a Point Cloud Using Scene Depth

Place points in the real-world using the scene's depth data to visualize the shape of the physical environment.  

## Overview

- Note: This sample code project is associated with WWDC20 session [10611: Explore ARKit 4](https://developer.apple.com/wwdc20/10611/).

## Configure the Sample Code Project

Before you run the sample code project in Xcode, set the run destination to an iPad Pro with a LiDAR sensor, running iPadOS 14.0 or later.

## To extract RGB, Depth and Point Cloud

pixelBuffer, depthMap : RGB와 Depth 정보를 정의한다.

코드 내에 기입된 Projection 함수(센서들에 의해 지정되는 matrix)를 적용하여 Point Cloud data를 생성한다.

각각의 2D 상의 pixel, Depth, 3D coordinates를 array로 저장한다.
