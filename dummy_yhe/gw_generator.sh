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

send "cd checkOut/phocnet_kws/experiments/seg_based/datasets\r"

send "python3\r"

send "from gw import GWDataset\r"

send "ds = GWDataset(gw_root_dir='/vol/corpora/document-image-analysis/gw')\r"

send "ds.mainLoader('train')\r"

send "word_img, embedding, class_id, is_query, transciption = ds.__getitem__(3)\r"

interact


