# %%
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import ast
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-g","--genes",help="gene list.",type=argparse.FileType('r'))
parser.add_argument("-o","--out",help="output file name.",type=str)
args = parser.parse_args()

with args.genes as f:
    genelist = f.read().splitlines()

genelist = [i for i in genelist if i]
print(f'Runnning LOF pipeline for {len(genelist)} genes')

# %%
mmrf_muse_path = "/home/vivek/mmrf_data/mmrf_variant_callers/hg19/muse/annovar/vcf_with_AF/vcf_reheader/fathmm/filtered_vcf/"
mmrf_mutect_path = "/home/vivek/mmrf_data/mmrf_variant_callers/hg19/mutect2/annovar/vcf_reheader/fathmm/filtered_vcf"
mmrf_sniper_path = "/home/vivek/mmrf_data/mmrf_variant_callers/hg19/somatic_sniper/annovar/vcf_with_AF/vcf_reheader/fathmm/filtered_vcf"
mmrf_varscan_path = "/home/vivek/mmrf_data/mmrf_variant_callers/hg19/varscan2/annovar/vcf_reheader/fathmm/filtered_vcf"

ega_muse_path = "/home/vivek/ega_data_1901/muse/fathmm_filtered/filtered_vcf"
ega_mutect_path = "/home/vivek/ega_data_1901/MUTECT2_analysis/fathmm_filtered/filtered_vcf"
ega_sniper_path = "/home/vivek/ega_data_1901/SNIPER_analysis/fathmm_filtered/filtered_vcf"
ega_varscan_path = "/home/vivek/ega_data_1901/varscan2/fathmm_filtered/filtered_vcf"

aiims_mm_muse_path = "/home/vivek/aiims_data_processing/muse/mm_sample_analysis/fathmm_filtered/filtered_vcf"
aiims_mm_mutect_path = "/home/vivek/aiims_data_processing/MUTECT2/mm_sample_analysis/fathmm_filtered/filtered_vcf"
aiims_mm_sniper_path = "/home/vivek/aiims_data_processing/SNIPER/mm_sample_analysis/fathmm_filtered/filtered_vcf"
aiims_mm_varscan_path = "/home/vivek/aiims_data_processing/varscan2/mm_annovar/vcf_reheader/fathmm/filtered_vcf"

aiims_mgus_muse_path = "/home/vivek/aiims_data_processing/muse/mgus_sample_analysis/fathmm_filtered/filtered_vcf"
aiims_mgus_mutect_path = "/home/vivek/aiims_data_processing/MUTECT2/mgus_sample_analysis/fathmm_filtered/filtered_vcf"
aiims_mgus_sniper_path = "/home/vivek/aiims_data_processing/SNIPER/mgus_sample_analysis/fathmm_filtered/filtered_vcf"
aiims_mgus_varscan_path = "/home/vivek/aiims_data_processing/varscan2/mgus_annovar/vcf_reheader/fathmm/filtered_vcf"


mmrf_sample_list = open('/home/vivek/mmrf_data/mmrf_variant_callers/mmrf_mm_sample_list.txt').read().split('\n')
mmrf_sample_list = [x for x in mmrf_sample_list if not "PB" in x]
aiims_mm_sample_list = open('/home/vivek/aiims_data_processing/scripts/mm_list.txt').read().split('\n')
aiims_mm_sample_list = [x for x in aiims_mm_sample_list if x != '']
aiims_mgus_sample_list = open('/home/vivek/aiims_data_processing/scripts/mgus_list.txt').read().split('\n')
aiims_mgus_sample_list = [x for x in aiims_mgus_sample_list if x != '']
ega_sample_list = open('/home/vivek/ega_data_1901/scripts/ega_mgus_list.txt').read().split('\n')
ega_sample_list = [x for x in ega_sample_list if x != '']
all_mm = list(set(mmrf_sample_list + aiims_mm_sample_list))
all_mm = [x for x in all_mm if x != '']
all_mgus = list(set(aiims_mgus_sample_list + ega_sample_list))
all_mgus = [x for x in all_mgus if x != '']

aiims_mgus_cnv_path = "/home/vivek/aiims_data_processing/cnvkit_analysis/mgus"
aiims_mm_cnv_path = "/home/vivek/aiims_data_processing/cnvkit_analysis/mm"
ega_cnv_path = "/home/vivek/ega_data_1901/cnvkit_analysis"
mmrf_cnv_path = "/home/vivek/mmrf_data/cnv_analysis/filtered_cns"


# %%
def read_sam_cnvs(sam, disease):
    if disease == "mgus":
        if 'SM' in sam:
            df = pd.read_csv(os.path.join(aiims_mgus_cnv_path, sam, sam+"BM_tumor.call.cns"), sep="\t")
            df = df[df["cn"]!= 2]
            df = df[df["gene"] != '-']
            df = df[~df["chromosome"].str.contains("random")]
            df = df[~df["chromosome"].str.contains("hap")]
            df = df[~df["chromosome"].str.contains("chrUn")].reset_index(drop=True)
        elif "CR" in sam:
            df = pd.read_csv(os.path.join(ega_cnv_path, sam, sam+"-BM_dedup.realigned.call.cns"), sep="\t")
            df = df[df["cn"]!= 2]
            df = df[df["gene"] != '-']
            df = df[~df["chromosome"].str.contains('GL')].reset_index(drop=True)
    elif disease == "mm":
        if 'SM' in sam:
            df = pd.read_csv(os.path.join(aiims_mm_cnv_path, sam, sam+"BM_tumor.call.cns"), sep="\t")
            df = df[df["cn"]!= 2]
            df = df[df["gene"] != '-']
            df = df[~df["chromosome"].str.contains("random")]
            df = df[~df["chromosome"].str.contains("hap")]
            df = df[~df["chromosome"].str.contains("chrUn")].reset_index(drop=True)
        if "MMRF" in sam:
            df = pd.read_csv(os.path.join(mmrf_cnv_path, sam+"_BM.cns"), sep="\t")            
            
    df['sample_id'] = [sam for i in range(df.shape[0])]
    return df

# %%
def filter_sam_cnvs(sam_df, gen):
    gene_idx = []
    for idx in range(sam_df.shape[0]):
        try:
            string = sam_df.iloc[idx]["gene"]
            genes = sorted(list(set(ast.literal_eval(string))))
        except:
            genes = sorted(list(set(sam_df.iloc[idx]["gene"].split(','))))
        
        if gen in genes:
            gene_idx.append(idx)
    
    return sam_df.iloc[gene_idx]

# %%
def all_samp_gene_cnv(gene_name, disease):
    df_gene_cnv = pd.DataFrame()
    if disease == "mm":
        for sam in all_mm:
            try:
                # read sample CNVs
                df = read_sam_cnvs(sam, disease)
                # Filter CNV for a given gene
                df = filter_sam_cnvs(df, gene_name)
                df_gene_cnv = pd.concat([df_gene_cnv, df]).reset_index(drop=True)
            except:
                continue
    elif disease == "mgus":
        for sam in all_mgus:
            try:
                # read sample CNVs
                df = read_sam_cnvs(sam, disease)
                # Filter CNV for a given gene
                df = filter_sam_cnvs(df, gene_name)
                df_gene_cnv = pd.concat([df_gene_cnv, df]).reset_index(drop=True)
            except:
                continue
            
    return df_gene_cnv


# %%
def get_gene_df(sampleid, disease, gene_name):
    
    if "MMRF" in sampleid:
        df_muse = pd.read_csv(os.path.join(mmrf_muse_path, sampleid + ".vcf"), sep="\t")
        df_mutect = pd.read_csv(os.path.join(mmrf_mutect_path, sampleid + ".vcf"), sep="\t")
        df_sniper = pd.read_csv(os.path.join(mmrf_sniper_path, sampleid + ".vcf"), sep="\t")
        df_varscan = pd.read_csv(os.path.join(mmrf_varscan_path, sampleid + ".vcf"), sep="\t")
        df = pd.concat([df_muse, df_mutect, df_sniper, df_varscan])
        df = df[df["Gene_refGene"] == gene_name]
        df = df.drop_duplicates(ignore_index=True)
        
    if "CR" in sampleid:
        df_muse = pd.read_csv(os.path.join(ega_muse_path, sampleid + ".vcf"), sep="\t")
        df_mutect = pd.read_csv(os.path.join(ega_mutect_path, sampleid + ".vcf"), sep="\t")
        df_sniper = pd.read_csv(os.path.join(ega_sniper_path, sampleid + ".vcf"), sep="\t")
        df_varscan = pd.read_csv(os.path.join(ega_varscan_path, sampleid + ".vcf"), sep="\t")
        df = pd.concat([df_muse, df_mutect, df_sniper, df_varscan])
        df = df[df["Gene_refGene"] == gene_name]
        df = df.drop_duplicates()    
    
    if "SM" in sampleid:
        if disease == "mm":
            df_muse = pd.read_csv(os.path.join(aiims_mm_muse_path, sampleid + ".vcf"), sep="\t")
            df_mutect = pd.read_csv(os.path.join(aiims_mm_mutect_path, sampleid + ".vcf"), sep="\t")
            df_sniper = pd.read_csv(os.path.join(aiims_mm_sniper_path, sampleid + ".vcf"), sep="\t")
            df_varscan = pd.read_csv(os.path.join(aiims_mm_varscan_path, sampleid + ".vcf"), sep="\t")
            df = pd.concat([df_muse, df_mutect, df_sniper, df_varscan])
            df = df[df["Gene_refGene"] == gene_name]
            df = df.drop_duplicates()
        elif disease == "mgus":
            df_muse = pd.read_csv(os.path.join(aiims_mgus_muse_path, sampleid + ".vcf"), sep="\t")
            df_mutect = pd.read_csv(os.path.join(aiims_mgus_mutect_path, sampleid + ".vcf"), sep="\t")
            df_sniper = pd.read_csv(os.path.join(aiims_mgus_sniper_path, sampleid + ".vcf"), sep="\t")
            df_varscan = pd.read_csv(os.path.join(aiims_mgus_varscan_path, sampleid + ".vcf"), sep="\t")
            df = pd.concat([df_muse, df_mutect, df_sniper, df_varscan])
            df = df[df["Gene_refGene"] == gene_name]
            df = df.drop_duplicates()
        
    df['sample_id'] = [sampleid for i in range(len(df))]
    
    return df
        
        
def get_gene_df_all_samples(gene_name, disease, save_out = False):
    
    df_gene = pd.DataFrame()
    df_gene1 = pd.DataFrame()
    mm_sample_count, mgus_sample_count = 0,0
    if disease == "mm":
        for sam in all_mm:
            df = get_gene_df(sam, "mm", gene_name)
            if df.shape[0] > 0:
                mm_sample_count += 1
                df_gene = pd.concat([df_gene, df])

        if df_gene.shape[0] > 0:
            df_gene["chr"] = [i.split("chr")[-1] for i in df_gene["chr"]]
            df_gene = df_gene.drop_duplicates(ignore_index=True)
            df_gene = df_gene[["sample_id","chr","pos","ref","alt","Gene_refGene","Func_refGene","Exonic_refGene","AF","AD"]]
            df_gene1 = df_gene
            # df_gene1 = df_gene.groupby(['pos']).agg(lambda x: list(x))
        
            if save_out:
                df_gene1.to_excel(os.path.join("/home/vivek/jupyter_notebooks/bio_dgi_extension/final_compilation_work/gene_snvs/ungrouped_sample_level/mm", gene_name + ".xlsx"), index=False)
        
        return df_gene1, mm_sample_count
        
    elif disease == "mgus":
        for sam in all_mgus:
            df = get_gene_df(sam, "mgus", gene_name)
            if df.shape[0] > 0:
                mgus_sample_count += 1
                df_gene = pd.concat([df_gene, df])

        if df_gene.shape[0] > 0:
            df_gene["chr"] = [i.split("chr")[-1] for i in df_gene["chr"]]
            df_gene = df_gene.drop_duplicates(ignore_index=True)
            df_gene = df_gene[["sample_id","chr","pos","ref","alt","Gene_refGene","Func_refGene","Exonic_refGene","AF","AD"]]
            df_gene1 = df_gene
            # df_gene1 = df_gene.groupby(['pos']).agg(lambda x: list(x))
    
            if save_out:
                file_path = os.path.join("/home/vivek/jupyter_notebooks/bio_dgi_extension/final_compilation_work/gene_snvs/", disease, gene_name+".xlsx")
                if not os.path.exists(file_path):
                    df_gene1.to_excel(file_path, index=False)
            
            # if save_out:
            #     df_gene1.to_excel(os.path.join("/home/vivek/jupyter_notebooks/bio_dgi_extension/final_compilation_work/gene_snvs/ungrouped_sample_level/mgus", gene_name + ".xlsx"), index=False)
            
        return df_gene1, mgus_sample_count
    

def get_gene_df_filtered(df_gene1, disease, gname, save_out = True):
    df_gene2 = pd.DataFrame()
    chr2, ref2,alt2, gene2, func2, exonic2, af2, ad2 = [],[],[],[],[],[],[],[]
    for idx in range(df_gene1.shape[0]):
        
        chr2.append(df_gene1.iloc[idx]["chr"][0])
        
        if isinstance(df_gene1.iloc[idx]["ref"], list):
            if list(set(df_gene1.iloc[idx]["ref"])).__len__() > 1:
                ref2.append('/'.join(list(set(df_gene1.iloc[idx]["ref"]))))
            else:
                ref2.append(df_gene1.iloc[idx]["ref"][0])
        else:
            ref2.append(df_gene1.iloc[idx]["ref"])
        
        if isinstance(df_gene1.iloc[idx]["alt"], list):
            if list(set(df_gene1.iloc[idx]["alt"])).__len__() > 1:
                alt2.append('/'.join(list(set(df_gene1.iloc[idx]["alt"]))))
            else:
                alt2.append(df_gene1.iloc[idx]["alt"][0])
        else:
            alt2.append(df_gene1.iloc[idx]["alt"])
        
        if isinstance(df_gene1.iloc[idx]["Gene_refGene"], list):
            gene2.append(df_gene1.iloc[idx]["Gene_refGene"][0])
        else:
            gene2.append(df_gene1.iloc[idx]["Gene_refGene"])
        
        if isinstance(df_gene1.iloc[idx]["Func_refGene"], list):
            if list(set(df_gene1.iloc[idx]["Func_refGene"])).__len__() > 1:
                func2.append('/'.join(list(set(df_gene1.iloc[idx]["Func_refGene"]))))
            else:
                func2.append(df_gene1.iloc[idx]["Func_refGene"][0])
        else:
            func2.append(df_gene1.iloc[idx]["Func_refGene"])
            
        if isinstance(df_gene1.iloc[idx]["Exonic_refGene"], list):
            if list(set(df_gene1.iloc[idx]["Exonic_refGene"])).__len__() > 1:
                    exonic2.append('/'.join(list(set(df_gene1.iloc[idx]["Exonic_refGene"]))))
            else:
                exonic2.append(df_gene1.iloc[idx]["Exonic_refGene"][0])
        else:
            exonic2.append(df_gene1.iloc[idx]["Exonic_refGene"])
            
        
        if isinstance(df_gene1.iloc[idx]["AF"], list):
            if df_gene1.iloc[idx]["AF"].__len__() > 1:
                print("this")
                # print(df_gene1.iloc[idx]["AF"])
                aaf = df_gene1.iloc[idx]["AF"]
                af2.append(np.median([float(aaf[i].split(',')[0]) if isinstance(aaf[i],str) else float(aaf[i]) for i in range(aaf.__len__())]))
            else:
                print("this0")
                if ',' in df_gene1.iloc[idx]["AF"]:
                    afs = [float(i) for i in df_gene1.iloc[idx]["AF"].split(',')]
                    af2.append(np.median(afs))
                else:
                    print("this00")
                    af2.append(df_gene1.iloc[idx]["AF"][0])
        else:
            af2.append(df_gene1.iloc[idx]["AF"])
            
        if isinstance(df_gene1.iloc[idx]["AD"], list):
            if df_gene1.iloc[idx]["AD"].__len__() > 1:
                aad = df_gene1.iloc[idx]["AD"]
                ad2.append(np.median([float(aad[i].split(',')[0]) if isinstance(aad[i],str) else float(aad[i]) for i in range(aad.__len__())]))
                # ad2.append(np.median([float(i) for i in df_gene1.iloc[idx]["AD"]]))
            else:
                if ',' in df_gene1.iloc[idx]["AD"]:
                    ads = [float(i) for i in df_gene1.iloc[idx]["AD"].split()]
                    ad2.append(np.median(ads))
                else:
                    ad2.append(df_gene1.iloc[idx]["AD"][0])
        else:
            ad2.append(df_gene1.iloc[idx]["AD"])
            
    df_gene2["chr"] = chr2
    df_gene2["pos"] = df_gene1.index.tolist()
    df_gene2["ref"] = ref2
    df_gene2["alt"] = alt2
    df_gene2["gene"] = gene2
    df_gene2["func"] = func2
    df_gene2["exonic"] = exonic2
    df_gene2["af"] = af2
    df_gene2["ad"] = ad2
    
    if save_out:
        file_path = os.path.join("/home/vivek/jupyter_notebooks/bio_dgi_extension/final_compilation_work/gene_snvs/", disease, gname+".xlsx")
        if not os.path.exists(file_path):
            df_gene2.to_excel(file_path, index=False)
    return df_gene2

def get_gene_processed_snvs(gene_name, disease):
    file_path = os.path.join("/home/vivek/jupyter_notebooks/bio_dgi_extension/final_compilation_work/gene_snvs/", disease, gene_name+".xlsx")
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df_mm, _ = get_gene_df_all_samples(gene_name, disease, save_out=False)
        df = get_gene_df_filtered(df_mm, disease, gene_name, save_out=True)
    return df

def filter_gene_snvs_in_cnvs(df_snvs, start_pos, end_pos):
    df_snvs = df_snvs.query('pos >='+ str(start_pos) +'  & pos <= ' + str(end_pos))
    return df_snvs    

# %%
# df = get_gene_processed_snvs("MYC", "mm")

# %%
chess_data = pd.read_csv('/home/vivek/jupyter_notebooks/bio_dgi_extension/farcastbio/chess_database/chess3.0.1.gff', sep='\t', header=None)
chess_data.columns = ['chr', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
chess_data['gene_id'] = chess_data['attributes'].apply(lambda x: x.split('gene_name')[-1].split(';')[0].split('=')[-1])
chess_data['id'] = chess_data['attributes'].apply(lambda x: x.split('ID')[-1].split(';')[0].split('=')[-1])
chess_data = chess_data.drop(["score"], axis=1)
# genelist = chess_data["gene_id"].unique().tolist()

# %%
def find_cds_deletion(chess_gene, start, end):
    # total_transcripts = chess_gene[chess_gene["type"] == "transcript"].shape[0]
    gene_unique_id = chess_gene["id"].unique().tolist()
    del_true = 0
    for uid in gene_unique_id:
        chess_gene_uid_cds = chess_gene[(chess_gene["id"] == uid) & (chess_gene["type"] == "CDS")]
        total_cds_len = sum([j-i for i,j in zip(chess_gene_uid_cds["start"].tolist(), chess_gene_uid_cds["end"].tolist())])
        deL_cds_len = sum([j-i for i,j in zip(chess_gene_uid_cds["start"].tolist(), chess_gene_uid_cds["end"].tolist()) if (i >= start) & (j <= end)])
        if deL_cds_len > total_cds_len/2:
            del_true += 1
        # else:
        #     del_false += 1
            
    if del_true == gene_unique_id.__len__():
        return True
    else:
        return False


# %%
def find_first_exon_deletion(chess_gene, start, end):
    gene_unique_id = chess_gene["id"].unique().tolist()
    del_true = 0
    for uid in gene_unique_id:
        chess_gene_uid_exon = chess_gene[(chess_gene["id"] == uid) & (chess_gene["type"] == "exon")].sort_values(by=['start'])
        chess_gene_uid_first_exon_start = chess_gene_uid_exon["start"].tolist()[0]
        chess_gene_uid_first_exon_end = chess_gene_uid_exon["end"].tolist()[0]
        if (chess_gene_uid_first_exon_start >= start) & (chess_gene_uid_first_exon_end <= end):
            del_true += 1
        # else:
        #     del_false += 1
            
    if del_true == gene_unique_id.__len__():
        return True
    else:
        return False

# %%
def find_splice_variants(chess_gene, start, end, snv_df):
    gene_unique_id = chess_gene["id"].unique().tolist()
    del_true = 0
    for uid in gene_unique_id:
        chess_gene_uid_exon = chess_gene[(chess_gene["id"] == uid) & (chess_gene["type"] == "transcript")].sort_values(by=['start'])
        chess_gene_uid_start = chess_gene_uid_exon["start"].tolist()[0]
        chess_gene_uid_end = chess_gene_uid_exon["end"].tolist()[0]
        snv_filt_df = filter_gene_snvs_in_cnvs(snv_df, chess_gene_uid_start, chess_gene_uid_start)
        if snv_filt_df.shape[0] > 0:
            snv_filt_df = snv_filt_df[snv_filt_df["func"].str.contians("splice")]
            if snv_filt_df.shape[0] > 0:
                del_true += 1

    if del_true == gene_unique_id.__len__():
        return "True"
    else:
        return "False"

# %%
def find_frameshift_variants(chess_gene, start, end, snv_df):
    gene_unique_id = chess_gene["id"].unique().tolist()
    del_true = 0
    for uid in gene_unique_id:
        chess_gene_uid_exon = chess_gene[(chess_gene["id"] == uid) & (chess_gene["type"] == "transcript")].sort_values(by=['start'])
        chess_gene_uid_start = chess_gene_uid_exon["start"].tolist()[0]
        chess_gene_uid_end = chess_gene_uid_exon["end"].tolist()[0]
        snv_filt_df = filter_gene_snvs_in_cnvs(snv_df, chess_gene_uid_start, chess_gene_uid_start)
        if snv_filt_df.shape[0] > 0:
            snv_filt_df = snv_filt_df[snv_filt_df["func"].str.contians("frameshift")]
            if snv_filt_df.shape[0] > 0:
                del_true += 1

    if del_true == gene_unique_id.__len__():
        return "True"
    else:
        return "False"

# %%
def lof_pipeline(gene_name, disease):
    lof_sample_list = []
    df_cnv = all_samp_gene_cnv(gene_name, disease)
    if df_cnv.shape[0] > 0:
        df_cnv_del = df_cnv[df_cnv["cn"] < 2]
        chess_gene = chess_data[chess_data["gene_id"] == gene_name]
        snv_df = get_gene_processed_snvs(gene_name, disease)
        if df_cnv_del.shape[0] > 0:
            for idx in range(df_cnv_del.shape[0]):
                sam_id = df_cnv_del.iloc[idx]["sample_id"]
                cnv_start = df_cnv_del.iloc[idx]["start"]
                cnv_end = df_cnv_del.iloc[idx]["end"]
                case_a = find_cds_deletion(chess_gene, cnv_start, cnv_end)
                case_b = find_first_exon_deletion(chess_gene, cnv_start, cnv_end)
                case_c = find_splice_variants(chess_gene, cnv_start, cnv_end, snv_df)
                case_d = find_frameshift_variants(chess_gene, cnv_start, cnv_end, snv_df)
                if case_a or case_b or case_c or case_d:
                    lof_sample_list.append(sam_id)
                    
            
    lof_sample_list = list(set(lof_sample_list))
    return lof_sample_list.__len__(), lof_sample_list

# %%
# lof_pipeline("MYC", "mm")

# %%
df_lof = pd.DataFrame()
lof_mm, lof_mgus = [], []
lof_mm_sample_list, lof_mgus_sample_list = [], []

for gen in tqdm(genelist):
    try:
        no_mm_sample, lof_mm_sam = lof_pipeline(gen, "mm")
        lof_mm.append(no_mm_sample)
        lof_mm_sample_list.append(lof_mm_sam)
    except:
        lof_mm.append(0)
        lof_mm_sample_list.append([])
    try:
        no_mgus_sample, lof_mgus_sam = lof_pipeline(gen, "mgus")
        lof_mgus.append(no_mgus_sample)
        lof_mgus_sample_list.append(lof_mgus_sam)
    except:
        lof_mgus.append(0)
        lof_mgus_sample_list.append([])
    
df_lof["gene_name"] = genelist
df_lof["lof_mm"] = lof_mm
df_lof["lof_mgus"] = lof_mgus
df_lof["lof_mm_sample_list"] = lof_mm_sample_list
df_lof["lof_mgus_sample_list"] = lof_mgus_sample_list
df_lof.to_excel("lof_"+args.out+".xlsx", index=False)
