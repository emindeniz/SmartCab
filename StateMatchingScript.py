import pandas as pd
import numpy as np
states = {'Light': ['red','red','red','red','red','red','red','red','red','green','green','green','green','green','green','green','green','green','None'],
                    'Traffic': ['Oncoming','Oncoming','Oncoming','left','left','left','right','right','right','Oncoming','Oncoming','Oncoming','left','left','left','right','right','right','None'],
                    'next_waypoint': ['forward','right','left','forward','right','left','forward','right','left','forward','right','left','forward','right','left','forward','right','left','None']}  
Qtable = np.zeros((18,4))
states = pd.DataFrame(states)
print states