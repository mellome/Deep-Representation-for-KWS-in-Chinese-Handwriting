#!/usr/local/bin/expect
set timeout 30

spawn ssh yhe@fincher.cs.tu-dortmund.de

expect "password:"
 
send "Mello1265894\r"

expect "*#"

send "ssh kronos\r"

expect "password:"

send "Mello1265894\r"

expect "*#"

# send "read -p 'please enter your purpose: ' PURPOSE"

send "cd ~/checkOut/phocnet_kws/experiments/seg_based/\r"

# send "nohup nice python3 casia_train.py --gpu_id 2 --display 10 &>casia.log &\r"
send "nohup nice python3 casia_train.py &>casia.log &\r"

send "tail -f casia.log\r"

interact
