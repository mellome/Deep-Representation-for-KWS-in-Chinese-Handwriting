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

send "cd ~/checkOut/phocnet_kws/experiments/seg_based/\r"

interact