import sys, time, struct
sys.path.insert(0,'.')
import numpy as np, core_part as c
rng=np.random.default_rng(1)

def test_image(n, seed=0):
    r=np.random.default_rng(seed)
    y,x=np.mgrid[0:n,0:n]
    img=((np.sin(x/(n/6))+np.cos(y/(n/9))+2)/4*255)
    img[n//4:n//2,n//4:n//2]=255
    img=img+r.normal(0,12,img.shape)           # texture so every grey level shows up
    return np.clip(img,0,255).astype(np.uint8)

def write_variant(path,z,fs,kind):
    x=np.stack([z.real,z.imag],1)
    if kind=='u8': d=np.clip(np.rint(x*127+128),0,255).astype(np.uint8); bits,tag=8,1
    elif kind=='i16': d=np.clip(np.rint(x*32767),-32768,32767).astype('<i2'); bits,tag=16,1
    elif kind=='i24':
        v=np.clip(np.rint(x*8388607),-8388608,8388607).astype(np.int32)
        d=np.stack([v&255,(v>>8)&255,(v>>16)&255],-1).astype(np.uint8); bits,tag=24,1
    elif kind=='i32': d=np.clip(np.rint(x*2147483647.0),-2147483648,2147483647).astype('<i4'); bits,tag=32,1
    elif kind=='f32': d=x.astype('<f4'); bits,tag=32,3
    elif kind=='f64': d=x.astype('<f8'); bits,tag=64,3
    raw=d.tobytes()
    with open(path,'wb') as f:
        f.write(b'RIFF'+struct.pack('<I',36+len(raw))+b'WAVE'+struct.pack('<4sIHHIIHH',b'fmt ',16,tag,2,fs,fs*bits//4,bits//4,bits)+b'data'+struct.pack('<I',len(raw))+raw)

def run(n,wpm,fmt='wav',tone=0.0,call="VE5KC",noise=0.0,convert=None,path='/tmp/t.wav'):
    px=test_image(n)
    text=c.pixels_to_text(px,n,call,wpm)
    plan=c.export_plan(text,wpm,fmt,5.0)
    assert plan['ok'],plan['err']
    c.write_iq_file(path,plan,tone)
    src=c.WavIQ(path)
    if convert or noise:
        z=src.read(0,src.n)
        if noise: z=z+(rng.normal(size=z.shape)+1j*rng.normal(size=z.shape)).astype(np.complex64)*noise
        write_variant(path+'.v.wav',z,src.fs,convert or 'i16'); src.close(); src=c.WavIQ(path+'.v.wav')
    t0=time.time()
    try: res=c.decode_source(src)
    except c.DecodeError as e:
        print(f"  FAIL n={n} wpm={wpm} {fmt} tone={tone} noise={noise} conv={convert}: {e}"); return False
    want=c.quantize_pixels(px)                          # TRUE quantised source image, not a re-parse of the text
    diff=int((res['pixels']!=want).sum())
    ok=diff==0
    print(f"  {'OK ' if ok else 'BAD'} n={n} wpm={wpm} {fmt} fs={plan['fs']} tone={tone} noise={noise} conv={convert} "
          f"call={res['callsign']} wpm_est={res['wpm']:.0f} diffpx={diff} dur={plan['duration']:.2f}s "
          f"marker={plan['marker_len']} pad={plan['pad']/plan['fs']:.3f}s dec={time.time()-t0:.2f}s")
    src.close(); return ok

if __name__=='__main__':
    ok=True
    print("speed sweep 20x20:")
    for w in (60,200,1000,5000,10000,20000,45000): ok&=run(20,w)
    print("RF64:")
    for w in (50000,100000,150000): ok&=run(20,w,'rf64')
    print("tone offsets:")
    for t in (-7000.0,1000.0,12345.6): ok&=run(20,500,tone=t)
    print("bit depths:")
    for k in ('u8','i16','i24','i32','f32','f64'): ok&=run(20,800,convert=k)
    print("resolutions:")
    ok&=run(40,3000); ok&=run(80,20000); ok&=run(120,10000); ok&=run(240,10000)
    print("light noise:")
    ok&=run(20,200,noise=0.2); ok&=run(20,2000,noise=0.3)
    print("ALL OK" if ok else "SOME FAILED")
