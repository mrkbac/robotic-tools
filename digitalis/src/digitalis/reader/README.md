# Reading Interface

There are two types of readers:

- Live: Connected to a live source, can't pause or seek. They display data in real-time as it is received.
  - Currently the Foxglove WebSocket Bridge is supported
- Static: Connected to a static source, can pause and seek.
  - Currently `.mcap` files are supported.
  - For broken `.mcap` files, it will update the file
