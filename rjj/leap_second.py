import numpy as np
from datetime import datetime
import requests

iers_url = 'https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list'
jan_1970 = 2208988800 # 1970 - 1900 in seconds
leapsec_fmt = '%Y-%m-%d %H:%M:60'

try:
    with open('leap-seconds.list') as list:
        for line in list:
            if line.startswith('#@'):
                exp_date_ntp = int(line[2:])
        ntp_now = datetime.now().timestamp() + jan_1970
        if exp_date_ntp < ntp_now:
            raise FileNotFoundError
except FileNotFoundError:
    with open('leap-seconds.list', 'w') as list:
        list.write(requests.get(iers_url).text)

leapsec_list = np.loadtxt('leap-seconds.list', dtype=int)

last_leapsec = datetime.utcfromtimestamp(leapsec_list[-1,0]-1-jan_1970)
last_leapsec = last_leapsec.strftime(leapsec_fmt)
print('The most recent leap second was', last_leapsec, 'UTC')
