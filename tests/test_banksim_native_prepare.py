import csv
import hashlib
import pytest
import numpy as np

from src.banksim_native_prepare import History, FEATURE_NAMES, prepare


def test_history_same_step_and_7step_expiry():
    h=History()
    base=dict(customer="c", merchant="m", category="'es_food'", amount=12.)
    a=h.vector(step=0, **base)
    assert len(a)==24
    assert a[16]==0 and a[21]==0
    h.observe(step=0,customer="c",merchant="m",amount=12.)
    b=h.vector(step=0, **base)
    assert b[16]==np.log1p(1)
    assert b[21]==np.log1p(1)
    assert b[19]==np.log1p(12)
    c=h.vector(step=7, **base)
    assert c[16]==np.log1p(1)
    assert c[21]==0
    assert c[22]==0 and c[23]==0


def test_label_blind_final_features_and_overwrite(tmp_path):
    header="step,customer,age,gender,zipcodeOri,merchant,zipMerchant,category,amount,fraud\n"
    lines=[
        "0,c1,20,F,z,m1,z,'es_food',10,0\n",
        "0,c1,20,F,z,m1,z,'es_food',20,1\n",
        "162,c2,20,F,z,m2,z,'es_food',30,DO_NOT_READ\n",
    ]
    source=tmp_path/"external.csv"
    source.write_text(header+"".join(lines))
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    dest=tmp_path/"output"
    meta=prepare(source,dest,expected_sha256=digest,expected_rows=3)
    assert meta["final_labels_accessed"] is False
    assert meta["partition_rows"]["train"]==2
    assert meta["partition_rows"]["final_test"]==1
    assert meta["development_labels_rows"]==2
    labels=np.load(dest/"development_labels.npy",allow_pickle=False)
    assert labels.tolist()==[0,1]
    X=np.load(dest/"features.npy",mmap_mode="r",allow_pickle=False)
    assert X.shape==(3,len(FEATURE_NAMES))
    assert X[0,16]==0
    assert X[1,16]==np.log1p(1)
    with pytest.raises(FileExistsError,match="overwrite"):
        prepare(source,dest,expected_sha256=digest,expected_rows=3)


def test_source_checksum_refusal(tmp_path):
    p=tmp_path/"s.csv"
    p.write_text("not approved")
    with pytest.raises(RuntimeError,match="checksum"):
        prepare(p,tmp_path/"out",expected_sha256="0"*64,expected_rows=1)
