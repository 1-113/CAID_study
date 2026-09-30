"""Section 4: generate six vector figures from saved experimental data."""
from pathlib import Path
import csv
import json
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'caid-study-mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'Code' else SCRIPT_DIR
DATA = SCRIPT_DIR / 'results'
FIG = ROOT / 'figure'
C = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.titlesize':9,
                     'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,
                     'legend.fontsize':7.5,'axes.linewidth':.6,'lines.linewidth':1.1,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300,
                     'axes.spines.top':False,'axes.spines.right':False})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG/name, metadata={'Creator':'Matplotlib; sec4_plot.py'})
    plt.close(fig)


def frame(ax, label, ylabel, xlabel='Normalized time'):
    ax.set_title(label, loc='left', fontweight='bold', pad=5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(alpha=.16, linewidth=.5)


def natural(include_sensitivity=True):
    traces = [json.loads(p.read_text()) for p in DATA.glob('natural_trace_*.json')]
    with (DATA/'natural_summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    def trace(**kwargs):
        return next(t for t in traces if all(t['config'].get(k)==v for k,v in kwargs.items()))
    def select(**kwargs):
        return [r for r in rows if all(r.get(k)==str(v) for k,v in kwargs.items())]
    fig,ax=plt.subplots(2,2,figsize=(140/25.4,4.6),layout='constrained')
    for m,c in ((.3,C[0]),(.7,C[1])):
        for game,ls in (('PD','-'),('SD','--')):
            q=trace(game=game,mean=m,benefit=5.,mechanism='reciprocal')
            t=np.array(q['t']);x=np.array(q['x'])
            ax[0,0].plot(t,x.min(axis=1),color=c,ls=ls,alpha=.7)
            ax[0,0].plot(t,x.max(axis=1),color=c,ls=ls,label=f'{game}, mean {m}')
            ax[0,1].semilogy(t,q['error'],color=c,ls=ls,label=f'{game}, mean {m}')
    ax[0,0].legend(ncol=2,frameon=False,loc='center right')
    ax[0,1].axhline(1e-3,color='.4',ls=':',lw=.8)
    frame(ax[0,0],'(A) Initial mean','State envelope')
    frame(ax[0,1],'(B) Whole-network error',r'$\|x-\gamma\mathbf{1}\|_\infty$')
    labels=['Current','No\npreference','Pairwise\npayoff']
    for j,game in enumerate(('PD','SD')):
        rr=select(study='mechanism',game=game,normalized=False)
        vals=[next(r for r in rr if r['mechanism']==m) for m in ('reciprocal','unbiased','original')]
        ax[1,0].errorbar(np.arange(3)+(-.08 if j==0 else .08),
            [float(r['arrival_mean']) for r in vals],
            yerr=[float(r['arrival_ci95_half']) for r in vals],fmt='os'[j],
            color=C[j],capsize=2,label=game)
        rr=select(study='mechanism',game=game)
        vals=[next(r for r in rr if r['mechanism']==m and r['normalized']==('False' if m=='reciprocal' else 'True'))
              for m in ('reciprocal','unbiased','original')]
        ax[1,1].errorbar(np.arange(3)+(-.08 if j==0 else .08),
            [float(r['arrival_mean']) for r in vals],
            yerr=[float(r['arrival_ci95_half']) for r in vals],fmt='os'[j],color=C[j],capsize=2,label=game)
    for a,label in ((ax[1,0],'(C) Original weight scales'),(ax[1,1],'(D) Matched total strength')):
        a.set_xticks(range(3),labels);a.set_ylim(bottom=0);a.legend(frameon=False)
        frame(a,label,r'Arrival time $T_{\mathrm{tol}}$',xlabel='Interaction rule')
    save(fig,'Fig1_natural.pdf')
    if not include_sensitivity:
        return
    fig,ax=plt.subplots(1,3,figsize=(140/25.4,2.25),layout='constrained')
    for a,param,sym in zip(ax,('epsilon','beta','alpha'),(r'$\epsilon$',r'$\beta$',r'$\alpha$')):
        rr=sorted(select(study='sensitivity_'+param),key=lambda r:float(r[param]))
        a.errorbar([float(r[param]) for r in rr],[float(r['arrival_mean']) for r in rr],
                   yerr=[float(r['arrival_ci95_half']) for r in rr],fmt='o-',color=C[0],capsize=2)
        frame(a,f'({chr(65+list(ax).index(a))}) {sym}',r'$T_{\mathrm{tol}}$',xlabel=sym)
    save(fig,'Fig2_sensitivity.pdf')


def faults():
    raw=json.loads((DATA/'fault_results.json').read_text())
    fig,ax=plt.subplots(2,2,figsize=(140/25.4,4.5),layout='constrained')
    names=['none','random','degree-neighbor','three-stage']
    labels=['No replacement','Random','No intensity filter','Three-stage']
    for k,(name,label) in enumerate(zip(names,labels)):
        q=np.load(DATA/f'fault_loss_{name}.npz')
        ax[0,0].semilogy(q['t'],q['error'],color=C[k],ls=[':','-.','--','-'][k],label=label)
    ax[0,0].axvline(5,color='.4',lw=.8);ax[0,0].axhline(1e-3,color='.5',lw=.8,ls=':')
    ax[0,0].legend(frameon=False)
    frame(ax[0,0],'(A) Loss of responsiveness',r'$\|x-x_p\mathbf{1}\|_\infty$')
    rows=[next(r for r in raw['summary'] if r['scenario']=='loss' and r['strategy']==n) for n in names]
    ax[0,1].errorbar(range(4),[r['recovery_mean'] for r in rows],
        yerr=[r['recovery_ci95'] for r in rows],fmt='o',color=C[0],capsize=2)
    ax[0,1].set_xticks(range(4),[label.replace(' ','\n') for label in labels])
    ax[0,1].set_xlim(-.5,3.5)
    frame(ax[0,1],'(B) Recovery across seeds',r'$T_{\mathrm{tol}}-5$',xlabel='Replacement rule')
    for k,scenario in enumerate(('reset','shortage','outage')):
        q=np.load(DATA/f'fault_{scenario}.npz')
        ax[1,0].semilogy(q['t'],q['error'],label=scenario.capitalize(),color=C[k],ls=['-','--','-.'][k])
        ax[1,1].step(q['t'],q['count'],where='post',label=scenario.capitalize(),color=C[k],ls=['-','--','-.'][k])
    for a in ax[1]:
        a.axvline(5,color='.6',lw=.7);a.axvline(15,color='.6',lw=.7,ls=':')
        a.legend(frameon=False)
    ax[1,1].set_yticks([0,1,3,5]);ax[1,1].set_xlim(0,30)
    frame(ax[1,0],'(C) Distinct recovery scenarios',r'$\|x-x_p\mathbf{1}\|_\infty$')
    frame(ax[1,1],'(D) Actual active-set size',r'$|\mathcal{S}(t)|$')
    save(fig,'Fig6_recovery.pdf')


def controls():
    def load(target,kind):
        return np.load(DATA/'control_trajectories'/f'control_main__BA50_seed0_target{target:g}_{kind}.npz')
    fig,ax=plt.subplots(3,2,figsize=(140/25.4,6.0),layout='constrained')
    for j,target in enumerate((.3,1.)):
        q=load(target,'nonlinear');t=q['t']
        ax[0,j].plot(t,q['x'],color=C[j],alpha=.28,lw=.65)
        ax[0,j].axhline(target,color='.25',ls='--',lw=.8)
        ax[1,j].semilogy(t,q['error'],color=C[j]);ax[1,j].axhline(1e-3,color='.5',ls=':',lw=.8)
        for k in range(q['u'].shape[1]):ax[2,j].plot(t,q['u'][:,k],color=C[k],lw=.85)
        ax[2,j].set_ylim(-.53,.53);ax[2,j].axhline(.5,color='.5',ls=':',lw=.8);ax[2,j].axhline(-.5,color='.5',ls=':',lw=.8)
        frame(ax[0,j],f'({chr(65+j)}) Target {target:g}','All node states')
        frame(ax[1,j],f'({chr(67+j)}) Tracking error',r'$\|x-x_p\mathbf{1}\|_\infty$')
        frame(ax[2,j],f'({chr(69+j)}) Applied inputs',r'$u_i$')
    save(fig,'Fig3_induction.pdf')
    fig,ax=plt.subplots(2,2,figsize=(140/25.4,4.5),layout='constrained')
    for j,target in enumerate((.3,1.)):
        for k,(kind,label) in enumerate((('nonlinear','Nonlinear'),('linear','Linear pinning'),('PI','Feasible PI'))):
            q=load(target,kind);t=q['t']
            ax[0,j].semilogy(t,q['error'],color=C[k],ls=['-','--','-.'][k],label=label)
            ax[1,j].plot(t,q['energy'],color=C[k],ls=['-','--','-.'][k],label=label)
        ax[0,j].legend(frameon=False);ax[0,j].axhline(1e-3,color='.5',ls=':',lw=.8)
        frame(ax[0,j],f'({chr(65+j)}) Target {target:g}',r'$\|x-x_p\mathbf{1}\|_\infty$')
        frame(ax[1,j],f'({chr(67+j)}) Accumulated energy',r'$\int_0^t\|u(s)\|^2\,ds$')
    save(fig,'Fig4_comparison.pdf')
    p=json.loads((DATA/'control_sensitivity.json').read_text())['summary']
    fig,ax=plt.subplots(2,2,figsize=(140/25.4,4.25),layout='constrained')
    gain=sorted([r for r in p if r['case'].startswith('gain')],key=lambda r:r['gain'])
    count=sorted([r for r in p if r['case'].startswith('L')]+[next(r for r in p if r['case']=='gain0.5')],key=lambda r:r['count'])
    for j,(rows,key,lab) in enumerate(((gain,'gain',r'Gain $\kappa$'),(count,'count',r'Controlled nodes $L$'))):
        for i,metric in enumerate(('stop_time','energy_to_stop')):
            av=[r[metric]['mean'] for r in rows];hw=[(r[metric]['ci95'][1]-r[metric]['ci95'][0])/2 for r in rows]
            ax[i,j].errorbar([r[key] for r in rows],av,yerr=hw,fmt='o-',color=C[j],capsize=2)
            frame(ax[i,j],f'({chr(65+2*i+j)}) '+('Time to stop' if i==0 else 'Control energy'),r'$T_s$' if i==0 else r'$E_u$',xlabel=lab)
        if j==0:
            ax[0,j].set_xticks([.125,.25,.5]);ax[1,j].set_xticks([.125,.25,.5])
        if j==1:
            ax[0,j].text(.14,.87,'5/10 reached',transform=ax[0,j].transAxes,fontsize=7.5)
            ax[0,j].set_xticks([1,3,5,10]);ax[1,j].set_xticks([1,3,5,10])
    save(fig,'Fig5_control_sensitivity.pdf')


if __name__=='__main__':
    natural()
    controls()
    faults()
