"""Compile the V24 documents with cross-PDF references from a clean source package.
Requires standard-library Python, XeLaTeX, BibTeX and the named fonts.
Initial passes create both label sets; no existing aux files are required.
"""
from pathlib import Path
import subprocess,shutil,json,datetime
def main():
    here=Path(__file__).resolve().parent;steps=[];counts={}
    exe=shutil.which('xelatex');bib=shutil.which('bibtex')
    if not exe or not bib:raise RuntimeError('XeLaTeX and BibTeX must be on PATH')
    flags=['-interaction=nonstopmode','-halt-on-error','-file-line-error']
    if 'miktex' in exe.lower():flags.insert(0,'-disable-installer')
    def run(stem,bibliography=False):
        cmd=[bib,stem] if bibliography else [exe,*flags,stem+'.tex']
        r=subprocess.run(cmd,cwd=here,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
        counts[stem]=counts.get(stem,0)+1
        (here/f'{stem}_compile_{counts[stem]}.txt').write_bytes(r.stdout)
        steps.append({'command':cmd,'exit_code':r.returncode})
        if r.returncode:raise RuntimeError(f'{stem} compile failed; inspect captured log')
    for stem in ['main_en','supplement_en']:run(stem);run(stem,True)
    for _ in range(2):
        for stem in ['main_en','supplement_en']:run(stem)
    run('reader_zh');run('reader_zh',True);run('reader_zh');run('reader_zh')
    (here/'compile_record.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steps':steps},indent=2),encoding='utf-8')
    print('Compiled three PDFs with freshly resolved cross-document labels.')
if __name__=='__main__':main()
