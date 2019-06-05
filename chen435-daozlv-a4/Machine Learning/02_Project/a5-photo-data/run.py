# 环境：macosx、stackless python & twisted
 
cmd = 'make-train-data.sh train-labels.txt > train-data.txt' # 全路径或者./相对路径
import subprocess
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
result = []
while p.poll() == None:
     line = p.stdout.readline()
     print(line) # 必须执行print，否则一直不返回，原因不明
      
