import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def render_chart(df, out_path: str):
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2, sharex=ax1)
    ax3 = fig.add_subplot(3,1,3, sharex=ax1)

    ax1.plot(df.index, df['close'], label='Close')
    ax1.plot(df.index, df['ema20'], label='EMA20')
    ax1.plot(df.index, df['ema50'], label='EMA50')
    ax1.plot(df.index, df['bb_upper'], label='BB Upper', linewidth=0.8)
    ax1.plot(df.index, df['bb_mid'], label='BB Mid', linewidth=0.8)
    ax1.plot(df.index, df['bb_lower'], label='BB Lower', linewidth=0.8)
    ax1.legend(loc='upper left')
    ax1.set_title('Price + EMA + BB')

    ax2.bar(df.index, df['volume'])
    ax2.plot(df.index, df['vol_sma20'])
    ax2.set_title('Volume')

    ax3.plot(df.index, df['rsi14'])
    ax3.axhline(70, linestyle='--', linewidth=0.8)
    ax3.axhline(50, linestyle='--', linewidth=0.8)
    ax3.axhline(30, linestyle='--', linewidth=0.8)
    ax3.set_title('RSI(14)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path
