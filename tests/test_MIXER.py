import anndata as ad
import numpy as np
import MIXER as mi


def test_load_example_adata():
    adata = mi.load_PBMC3k()
    assert adata is not None
    assert isinstance(adata, ad.AnnData)


def test_single_cell_GeneClust():
    raw_adata = mi.load_PBMC3k()
    info, genes_fast = mi.GeneClust(
        raw_adata, n_var_clusters=200, version="fast", verbosity=2, return_info=True
    )
    info, genes_ps = mi.GeneClust(
        raw_adata,
        n_obs_clusters=7,
        relevant_gene_pct=5,
        version="ps",
        verbosity=2,
        return_info=True,
    )
    assert genes_fast.shape[0] > 0 and genes_ps.shape[0] > 0
    assert np.all(info.varp["redundancy"] == info.varp["redundancy"].T)


def test_spatial_GeneClust():
    adata, img = mi.load_mouse_brain()
    info, genes_ps = mi.GeneClust(
        adata,
        img,
        n_obs_clusters=5,
        version="ps",
        modality="st",
        relevant_gene_pct=5,
        verbosity=2,
        return_info=True,
    )
    assert genes_ps.shape[0] > 0


def test_MIXER():
    adata_rna = mi.load_mouse_cortex()
    adata_st, img = mi.load_mouse_brain()

    params = {
        "version": "ps",
        "verbosity": 0,
        "relevant_gene_pct": 10,
        "return_info": True,
    }

    info_rna, sc_genes = mi.GeneClust(
        adata_rna, n_obs_clusters=23, modality="sc", **params
    )
    info_st, st_genes = mi.GeneClust(
        adata_st, img, n_obs_clusters=15, modality="st", alpha=0.7, **params
    )
    integration_info, integrated_genes = mi.integrate(
        adata_rna=info_rna, adata_st=info_st, return_info=True
    )

    assert integrated_genes.shape[0] > 0