"""Ball-free player pooling and geometry reliability; no labels enter either API."""
import warnings
import numpy as np
from mtah.phase4 import velocity


def player_motion_features(players, teams, fps=25, pitch=(105,68)):
    players, teams = np.asarray(players), np.asarray(teams)
    if players.ndim != 3 or players.shape[1:] != (len(teams),2) or not all(np.any(teams==s) for s in ('H','A')):
        raise ValueError('INVALID_MOTION_INPUT')
    v = velocity(players,fps)
    speed = np.linalg.norm(v,axis=-1)
    v[speed>12] = np.nan
    speed[speed>12] = np.nan
    pools = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for side in ('H','A'):
            mask = teams==side
            xy, vel, s = players[:,mask], v[:,mask], speed[:,mask]
            valid = np.isfinite(xy).all(axis=-1)
            count = valid.sum(axis=1)
            scaled = xy/np.array(pitch)
            features = [np.nanmean(scaled,axis=1), np.nanstd(scaled,axis=1),
                np.nanquantile(scaled,0.1,axis=1),np.nanquantile(scaled,0.9,axis=1),
                np.nanmean(vel,axis=1)/12,
                np.nanmedian(s,axis=1)[:,None]/12,np.nanquantile(s,.75,axis=1)[:,None]/12,
                (count/11)[:,None], ((s>1).sum(axis=1)/np.maximum(count,1))[:,None],
                ((s>3).sum(axis=1)/np.maximum(count,1))[:,None]]
            for x in range(4):
                for y in range(3):
                    inside = valid & (scaled[:,:,0]>=x/4) & (scaled[:,:,0]<(x+1)/4) & (
                        scaled[:,:,1]>=y/3) & (scaled[:,:,1]<(y+1)/3)
                    features.append((inside.sum(axis=1)/11)[:,None])
            raw = np.column_stack(features)
            known = np.isfinite(raw).astype(np.float32)
            pools.append(np.column_stack((np.clip(np.nan_to_num(raw),-2,2),known)))
    h,a = pools
    return np.column_stack((h-a,h+a)).astype(np.float32)


def geometry_reliability(ball,fps=25):
    ball = np.asarray(ball,dtype=float)
    if ball.ndim!=2 or ball.shape[1]!=2 or fps<=0:
        raise ValueError('INVALID_RELIABILITY_INPUT')
    n = len(ball); lag = max(1,round(.08*fps))
    visible = np.isfinite(ball).all(axis=1)
    residual, speed = np.full(n,np.nan), np.full(n,np.nan)
    if n>2*lag:
        residual[lag:-lag] = np.linalg.norm(ball[lag:-lag]-(ball[:-2*lag]+ball[2*lag:])/2,axis=1)
        speed[lag:-lag] = np.linalg.norm(ball[2*lag:]-ball[:-2*lag],axis=1)*fps/(2*lag)
    quality = np.full(n,0.5)
    known = visible & np.isfinite(residual) & np.isfinite(speed)
    quality[known] = np.exp(-np.minimum((residual[known]/3)**2,100))*np.exp(
        -np.minimum((np.maximum(speed[known]-45,0)/15)**2,100))
    quality[~visible] = 0
    return np.clip(quality,0,1).astype(np.float32)


def apply_policy(probabilities,hazard,reliability,variant,motion_home=None,max_weight=.35,logit_scale=.5):
    p = np.asarray(probabilities,dtype=float)
    q = np.asarray(reliability,dtype=float)
    hazard = np.asarray(hazard,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or q.shape!=(len(p),) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('INVALID_FUSION_INPUT')
    if hazard.shape!=(len(p),) or not np.isfinite(hazard).all() or (hazard<0).any() or (hazard>1).any() or (p<0).any():
        raise ValueError('INVALID_FUSION_INPUT')
    if (q<0).any() or (q>1).any() or not 0<=max_weight<=1 or not 0<logit_scale<=1:
        raise ValueError('INVALID_FUSION_PARAMETERS')
    if variant=='phase4': return p.copy(),np.asarray(hazard).copy()
    if variant not in ('reliable_boundary','reliable_tempered','motion_fusion'):
        raise ValueError('UNKNOWN_PHASE5_POLICY')
    effective_hazard = np.asarray(hazard)*q
    if variant=='reliable_boundary': return p.copy(),effective_hazard
    live = p[:,:2].sum(axis=1)
    home = np.divide(p[:,0],live,out=np.full(len(p),.5),where=live>0)
    logits = np.log(np.clip(home,1e-6,1-1e-6)/np.clip(1-home,1e-6,1))
    home = 1/(1+np.exp(-logits*(logit_scale+(1-logit_scale)*q)))
    if variant=='motion_fusion':
        m = np.asarray(motion_home)
        if m.shape!=(len(p),) or not np.isfinite(m).all() or (m<0).any() or (m>1).any():
            raise ValueError('INVALID_MOTION_PROBABILITY')
        weight = max_weight*(1-q)
        home = (1-weight)*home+weight*m
    return np.column_stack((live*home,live*(1-home),p[:,2])),effective_hazard
