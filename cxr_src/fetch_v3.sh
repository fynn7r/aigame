cd /data/cxr_first
curl -s -m 300 -X POST --data-binary @v3.csv 'http://10.0.79.98:8765/upload?name=v3.csv'
echo " csv=$?"
curl -s -m 120 -X POST --data-binary @/root/_dsh/v3.log 'http://10.0.79.98:8765/upload?name=v3.log'
echo " log=$?"
md5sum v3.csv
ls -la v3.csv