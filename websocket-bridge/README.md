# websocket-bridge

A Python library implementing the Foxglove WebSocket protocol for streaming robotics data.

## Overview

`websocket-bridge` provides client and server implementations of the [Foxglove WebSocket protocol](https://github.com/foxglove/foxglove-sdk/tree/main/ros/src/foxglove_bridge), enabling real-time communication between robotics applications and visualization tools like Foxglove Studio.

## Features

- **WebSocketBridgeServer**: Async server for publishing robotics data over WebSocket
- **WebSocketBridgeClient**: Async client for subscribing to data streams
- Full support for Foxglove WebSocket protocol messages (advertise, subscribe, publish, etc.)
