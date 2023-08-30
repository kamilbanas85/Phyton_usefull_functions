## Bar plot with adjusted thicks

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def create_bar_stacked_plot(df_ts):     
    
    fig, ax = plt.subplots()
             
    df_ts.plot( kind='bar', stacked=True, ax=ax)
                        
    # Make most of the ticklabels empty so the labels don't get too crowded
    ticklabels = ['']*len(df_ts.index)
    # Every 4th ticklable shows the month and day
    # Every 12th ticklabel includes the year
    # ticklabels[::4] = [item.strftime('%b %d') for item in df_ts.index[::4]]
    # ticklabels[::12] = [item.strftime('%b %d\n%Y') for item in df_ts.index[::12]]
    ticklabels[::12] = [item.strftime('%b \n%Y') for item in df_ts.index[::12]]
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=0)
    #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
    #            mode="expand", borderaxespad=0, ncol=3)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.xlabel('date', fontsize=20) 
    plt.subplots_adjust(right=0.8)
    plt.title('LNG demand', fontsize=20)
    #ax.get_yaxis().get_offset_text().set_position((-0.025,0))
    
    plt.show()      
