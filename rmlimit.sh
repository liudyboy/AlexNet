#! /bin/bash

NETCARD=enp6s18

tc qdisc del dev $NETCARD root handle 1
