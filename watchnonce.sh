pid="$1"
tail -n 100 pong.txt | egrep "pid${pid}_" | sort -n | tail -1 | egrep "time[0-9]+" -o | egrep "[0-9]+" -o
