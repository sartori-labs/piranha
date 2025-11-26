#!/bin/bash

# Runs Piranha clients locally on 3 different GPUs
./piranha-debug -p 0 -c files/samples/localhost_config.json --gtest_filter=EvalTest*2PC* >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=2 ./piranha -p 2 -c files/samples/localhost_config.json >/dev/null &
./piranha-debug -p 1 -c files/samples/localhost_config.json --gtest_filter=EvalTest*2PC*

