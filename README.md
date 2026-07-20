# F1 Telemetry Anamoly Pipeline

## 1.Project Overview

- The Problem: Telemetry failures in sensors of F1 cars are difficult to isolate and diagnose because the sensors are deeply intterconnected. A single fault can create cascading alarms across all sensors.
- The Solution: This pipeline utilises a 2 stage architecture to monitor telemetry in real-time. By injecting a wide variety of synthetic faults during training, and then passing it through a highly compressed TCN , we can isolate the fault.

## 2.Architecture
