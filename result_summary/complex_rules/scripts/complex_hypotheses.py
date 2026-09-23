"""Literature-selected checks, kept separate from automatic discovery."""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

import complex_investigation as ci
import differential_stats as ds


def definition(center, context, targets):
    ant = tuple(sorted([center+'_CENTER', *[c+'_NEIGHBOR' for c in context]]))
    con = tuple(sorted(c+'_NEIGHBOR' for c in targets))
    return ci.cell_rule_name(ant,con)


def known_specs():
    """Each center sees the other types; contextual arrows also exchange source/target."""
    registry = json.loads((ci.ROOT/'literature_niches.json').read_text())
    specs = []
    for niche in registry:
        types = niche['types']
        a,b = niche['context_pair']
        context = [t for t in types if t not in (a,b)]
        for organ in niche['organs']:
            for center in types:
                specs.append(dict(niche,organ=organ,rule=definition(center,[],[t for t in types if t!=center]),
                                  family='centered',kind='attracts',contrast='Control–Severe'))
            for source,target in [(a,b),(b,a)]:
                specs.append(dict(niche,organ=organ,rule=definition(source,context,[target]),
                                  family='contextual',kind='attracts',contrast='Control–Severe'))
    return pd.DataFrame(specs)


def paper_specs():
    """Atlas claims choose every tested type and direction, not strongest mined results."""
    source = 'https://doi.org/10.1126/scitranslmed.adu6032'
    claims = [
        (['Plasma','CD4T','Epithelial'],ci.ORGANS,
         'Atlas Fig. 2I–K and supplementary Fig. S3I–J: plasma cells and CD4 T cells occupy reproducible zones near the epithelium; test the plasma-centered epithelial/CD4 neighborhood across stages.',
         'The atlas did not test this three-item ARM rule; this is a claim-driven spatial extension, not proof of cell loss.'),
        (['Goblet','CD4T','Macrophage'],ci.ORGANS,
         'Atlas Fig. 2D–F and supplementary Fig. S3H: CD4 T cells and macrophages favor lamina propria over epithelium; test goblet-centered exclusion of their joint neighborhood across stages.',
         'The paper does not claim this exact joint avoidance; broad spatial alignment only.'),
        (['Endocrine','Epithelial','Muscle'],['Duodenum'],
         'Atlas Fig. 3I–K: endocrine cells localize closer to muscularis mucosa after transplantation; test the muscle-adjacent epithelial niche.',
         'Local association cannot establish endocrine-cell replacement or differentiation.'),
        (['Paneth','Epithelial','Muscle'],['Duodenum'],
         'Atlas Fig. 2B and supplementary Fig. S5: Paneth cells occupy the crypt base near proliferative epithelium; test its muscle-adjacent organization.',
         'Keep spatial organization separate from the uninteresting abundance-loss result; no stem-cell identity is measured.')]
    specs=[]
    for (a,b,c),organs,claim,caveat in claims:
        for organ in organs:
            for center,context,target in [(a,[],[b,c]), (c,[b],[a])]:
                specs.append(dict(organ=organ,rule=definition(center,context,target),types=[a,b,c],
                                  claim=claim,caveat=caveat,source=source,kind='avoids' if a=='Goblet' else 'attracts',
                                  contrast='Control–Severe'))
    return pd.DataFrame(specs)


def extra_definitions(specs):
    return dict(zip(specs.rule,specs.types))


def direction_audit(specs, analysis, rules, cells, metadata):
    """Paired patient sign tests: informative and all passing complex occurrences.

    All centered alternatives and each contextual reverse are tested. One BH family
    covers every niche, organ, stage and both endpoints; zero differences are ties.
    """
    config = analysis['config']
    raw = ci.investigation_rows(rules,config,informative=False)
    raw_states,_ = ci.matrices(rules,raw,cells,metadata,config,extra_definitions(specs),
                               fields=analysis['fields'])
    results=[]
    for (niche,organ,family), group in specs.groupby(['niche','organ','family'],sort=False):
        for a,b in combinations(group.rule,2):
            for stage in ['All',*ci.STAGES]:
                meta=metadata[metadata.Organ.eq(organ)]
                if stage!='All':
                    meta=meta[meta[config.score].eq(stage)]
                for endpoint,states in [('informative',analysis['states']),('all passing',raw_states)]:
                    eligible=analysis['eligible'].loc[a] & analysis['eligible'].loc[b]
                    delta=(states.loc[a].eq(1).astype(float)-states.loc[b].eq(1).astype(float)).where(eligible)
                    biopsy=ds.aggregate_fovs(delta.to_frame('delta').T,meta,'Biopsy')
                    bm=meta.drop_duplicates('Biopsy').drop(columns='FOV').rename(columns={'Biopsy':'FOV'})
                    patients=ds.aggregate_fovs(biopsy,bm,'PatientID').loc['delta'].dropna()
                    pos,neg=int((patients>1e-12).sum()),int((patients< -1e-12).sum())
                    p=binomtest(pos,pos+neg,.5).pvalue if pos+neg>=5 else np.nan
                    results.append(dict(niche=niche,organ=organ,family=family,stage=stage,endpoint=endpoint,
                                        first=a,second=b,patients=len(patients),positive=pos,negative=neg,
                                        mean_difference_pp=100*patients.mean(),p_value=p))
    result=pd.DataFrame(results)
    result['fdr']=np.nan
    valid=result.p_value.dropna()
    if len(valid):
        result.loc[valid.index,'fdr']=multipletests(valid,method='fdr_bh')[1]
    return result
