from common.paths import C1, C2, C5, C6
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import hilbert, welch, spectrogram

"""
I have 6 conditions in this experiment. 
naming convention: 
    {cond}_t{trial_no}_p{participant_no}
    solo:  cond 1 - solo (don't see one another, told "make interesting movements")
    spont: cond 2 - spontaneous sync (told same as 1, can see each other)
    self:  cond 3 - focus on yourself. Each player focuses on their own movements.
    other: cond 4 - focus on the other player's movements
    p1:    cond 5 - both people focus on player 1.
    p2:    cond 6 - both people focus on player 2.


Following guide in Pikovsky, Rosenblum & Kurths Appendix A2
scipy.signal.hilbert returns zeta(t) = s(t) + i s_H(t) = A(t)e^{i \phi(t)}

"""

def calculate_phase(df: pd.DataFrame):
    df.columns = ['id1', 'id2', 'one', 'x', 'y', 'z', 'two', 'r'] 
    df['zeta'] = hilbert(df['r'])
    df['zeta_demeaned'] = hilbert(df['r'] - df['r'].mean())
    
    #N.B. This next line is incorrect as of 3/9/26 @ 16:50, unwrap needs to take the band-passed version
    # df['theta'] = np.unwrap(np.angle(df['H']))
    return df


if __name__ == "__main__":
    df1 = pd.read_csv(f'{C5}/pair_05/p_05A_trial_01.csv', header=None)
    df2 = pd.read_csv(f'{C5}/pair_05/p_05B_trial_01.csv', header=None)
    # df2 = pd.read_csv(f'{C5}/pair_01/p_01B_trial_01.csv', header=None)
    df1.columns = ['id1', 'id2', 'one', 'x', 'y', 'z', 'two', 'r'] 
    df2.columns = ['id1', 'id2', 'one', 'x', 'y', 'z', 'two', 'r'] 
    # Check that last column is actually the overall distance, check shape
    print(df1.head())
    print(df1.shape)
    # last column is overall distance. Shape lines up. ~7200 rows = 30s @ 240Hz


    # apply hilbert transform to the 'r' column, unwrap into phase and compare
    # N.B. can't unwrap this part, need more pre-processing (see synchrony book appendix)
    
    df1 = calculate_phase(df1)
    print(df1.head())
    print(df1['r'].describe())
    
    # The data going into Hilbert needs to oscillate about zero, so will need to demean
    # plot the real and imaginary parts of the result of the hilbert transform
    
    # fig, ax = plt.subplots()
    # ax.plot(np.real(df1["zeta"]), np.imag(df1["zeta"]))                      # y against the index
    # ax.set_title('zeta (raw)')
    # ax.set_xlabel("s(t)")
    # ax.set_ylabel("s_H(t)")
    # plt.legend()
    # plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.real(df1["zeta_demeaned"])[720:], np.imag(df1["zeta_demeaned"])[720:])                      # y against the index
    ax.set_title('zeta (demeaned)')
    ax.set_xlabel("s(t)")
    ax.set_ylabel("s_H(t)")
    plt.legend()
    plt.show()


    # Looking at the ooutput of this following function is supposed to help me understand what band pass is needed.
    # 
    f, P = welch(np.array(df1['r'] - df1['r'].mean()), fs=240, nperseg=2048)
    f0 = f[np.argmax(P)]

    fig, ax = plt.subplots()
    ax.semilogy(f, P)
    ax.set_xlim(0, 3)
    ax.set_ylim(1e-3, 10)
    ax.axvline(f0, ls="--", color="k")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    plt.show()

    # f, t, S = spectrogram(df1['r'] - df1['r'].mean(), fs=240, nperseg=1200, noverlap=1100)
    # plt.pcolormesh(t, f, np.log10(S + 1e-12), shading="auto"); 
    # plt.ylim(0, 3)
    # plt.show()
    # df2 = calculate_phase(df2)

    # plotting each pair out of x,y,z 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    df1 = df1.iloc[240:]
    df2 = df2.iloc[240:]
    t = np.arange(len(df1)) / 239
    sc = ax.scatter(df1.x, df1.y, df1.z, c=t, cmap="RdYlGn_r", s=1)
    # ax.scatter(df2.x, -1 * df2.y[0:240], df2.z[0:240], c=t[0:240], cmap='RdYlGn_r', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.colorbar(sc, label="time (s)")
    plt.show()
    fig, axes = plt.subplots(2, 3)
    axes[0, 0].scatter(df1.x, df1.y, c=t, cmap="RdYlGn_r", s=1)
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 1].scatter(df1.x, df1.z, c=t, cmap="RdYlGn_r", s=1)
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    sc = axes[0, 2].scatter(df1.y, df1.z, c=t, cmap="RdYlGn_r", s=1)
    axes[0, 2].set_xlabel('Y')
    axes[0, 2].set_ylabel('Z')
    # plt.colorbar(sc, label="time (s)")
    axes[1, 0].scatter(df2.x, df2.y, c=t, cmap="RdYlGn_r", s=1)
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 1].scatter(df2.x, df2.z, c=t, cmap="RdYlGn_r", s=1)
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Z')
    sc = axes[1, 2].scatter(df2.y, df2.z, c=t, cmap="RdYlGn_r", s=1)
    axes[1, 2].set_xlabel('Y')
    axes[1, 2].set_ylabel('Z')
    plt.colorbar(sc, label="time (s)")
    plt.tight_layout()
    # axes[0].set_aspect('equal')
    # axes[1].set_aspect('equal')
    # axes[2].set_aspect('equal')
    plt.show()
    # fig, ax = plt.subplots()
    # ax.plot(df1["theta"], label= 'A - theta')                      # y against the index
    # ax.plot(df2["theta"], label= 'B - theta')
    # ax.set_title('A, B focuses on A')
    # ax.set_xlabel("time (ms)")
    # ax.set_ylabel("x")
    # plt.legend()
    # plt.show()

    # df1 = pd.read_csv(f'{C6}/pair_01/p_01A_trial_01.csv', header=None)
    # df2 = pd.read_csv(f'{C6}/pair_01/p_01B_trial_01.csv', header=None)
    # df1 = calculate_phase(df1)
    # df2 = calculate_phase(df2)

    # fig, ax = plt.subplots()
    # ax.plot(df1["theta"], label= 'A - theta')                      # y against the index
    # ax.plot(df2["theta"], label= 'B - theta')
    # ax.set_title('A,B focus on B')
    # ax.set_xlabel("time (ms)")
    # ax.set_ylabel("x")
    # plt.legend()
    # plt.show()
