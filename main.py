import streamlit as st
import numpy as np
from df.enhance import enhance, init_df
import soundfile as sf
import torchaudio
import matplotlib.pyplot as plt
import librosa
import torch
import tempfile
import os
import subprocess
import shutil
import time

import audio_core as ac

# --- Professional UI Setup (Clean White) ---
st.set_page_config(
    page_title="Audio Cleaner",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Minimal CSS to complement Streamlit's native Light theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Jost:wght@300;400;500;600&display=swap');
    
    .stApp {
        font-family: 'Jost', sans-serif;
    }

    h1 {
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem !important;
    }

    /* Hide unneeded elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Subtle button hover */
    div.stButton > button {
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)

st.title("Audio Cleaner")
st.markdown("专业级深度音频降噪与修复工具")

# --- Settings Section ---
with st.expander("⚙️ 降噪参数调节 (点击展开)", expanded=False):
    st.markdown("如果发现处理后声音发闷、失真或被吞音，可通过降低“降噪强度”或切换“引擎版本”来缓解。")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        atten_lim_db_val = st.slider(
            "降噪强度 (dB)", 
            min_value=5, 
            max_value=100, 
            value=100,
            help="值越小，保留的自然声音（和底噪）越多，失真越小；值越大，降噪越彻底。遇到明显失真时，建议降至 10-20 dB 尝试。"
        )
        actual_atten_lim = atten_lim_db_val if atten_lim_db_val < 100 else None
        
        selected_model = st.selectbox(
            "神经引擎版本",
            options=["DeepFilterNet3", "DeepFilterNet2"],
            index=0,
            help="DeepFilterNet3 降噪更强但少数情况可能失真；遇到严重失真可尝试切换至第二代引擎。"
        )
        
    with col_s2:
        st.write("") # Spacer
        st.write("") # Spacer
        use_post_filter = st.checkbox(
            "开启后处理滤波器", 
            value=False,
            help="启用后处理以消除残留细微噪音。开启可能会加重失真，非极端噪音情况建议关闭以保护音质。"
        )
        
        st.write("") # Spacer
        mask_only = st.checkbox(
            "仅提取掩蔽信号 (专业)", 
            value=False,
            help="[高级功能] 输出的不再是音频，而是分离出来的底层频率声学遮罩掩码信号，仅供专业音频工程师二次混音使用。"
        )

# --- Logic Layer ---

uploaded_file = st.file_uploader("", type=["mp3", "m4a", "wav", "flac", "ogg", "mov"], label_visibility="collapsed")

if uploaded_file:
    is_video = uploaded_file.name.lower().endswith('.mov')

    # 文件大小硬上限（200 MB 默认）— 防止 OOM 与上传时间过长
    try:
        ac.validate_size(len(uploaded_file.getbuffer()))
    except ac.FileTooLargeError as e:
        st.error(f"❌ {e}")
        st.stop()

    # 唯一临时路径（uuid4 替代 int(time.time())，避免同秒并发冲突）
    paths = ac.make_temp_paths(uploaded_file.name)
    input_path = str(paths.input_path)
    work_wav_path = str(paths.work_wav_path)

    # 1. Processing Pipeline
    with st.status("正在准备素材...", expanded=False) as status:
        try:
            ac.check_ffmpeg()  # 缺 ffmpeg → 立即报错而不是后面白屏

            # Save to disk directly from buffer
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Get duration safely via ffprobe (with librosa fallback + 错误暴露)
            duration = ac.probe_duration(input_path)

            # 时长硬上限
            ac.validate_duration(duration)

            # Convert to working 48kHz mono WAV via memory-safe ffmpeg
            ac.convert_to_work_wav(input_path, work_wav_path)
        except ac.FFmpegMissingError as e:
            status.update(label="环境错误", state="error")
            st.error(f"❌ ffmpeg 未安装或不在 PATH：{e}")
            st.stop()
        except ac.FFmpegFailedError as e:
            status.update(label="处理失败", state="error")
            st.error(f"❌ ffmpeg 处理失败：{e}")
            paths.cleanup()
            st.stop()
        except ac.AudioTooLongError as e:
            status.update(label="音频超长", state="error")
            st.error(f"❌ {e}")
            paths.cleanup()
            st.stop()
        except Exception as e:
            status.update(label="未知错误", state="error")
            st.error(f"❌ 未预期错误：{type(e).__name__}: {e}")
            paths.cleanup()
            st.stop()
        status.update(label="准备就绪", state="complete")

    # 2. Hero Section: Preview
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption("原始采样预览")
        # 传递磁盘路径而非内存对象，允许 Streamlit 流式加载，避免大文件预览失败
        if is_video:
            st.video(input_path)
        else:
            st.audio(input_path)
            
    with col2:
        st.metric("音频时长", f"{duration:.1f}s")
        st.metric("原始体积", f"{uploaded_file.size/1024/1024:.1f} MB")

    st.divider()

    # 3. Processing Action
    if st.button(f"开始修复 ({selected_model})", type="primary"):
        start_t = time.time()

        out_wav_path = str(paths.out_wav_path)

        with st.spinner("正在加载 ARM64 神经引擎..."):
            model, df_state, _ = init_df(
                post_filter=use_post_filter,
                default_model=selected_model,
                mask_only=mask_only
            )

        waveform, sr = torchaudio.load(work_wav_path, backend='soundfile')

        # Chunked computation (Memory safe)
        chunk_len = ac.DEFAULT_CHUNK_SECONDS * sr
        n_chunks = int(np.ceil(waveform.shape[1] / chunk_len))

        p_text = st.empty()
        p_bar = st.progress(0)

        plot_orig = None
        plot_enh = None

        try:
            # 使用流式写入代替将整个文件保存在内存中，彻底解决内存溢出白屏问题
            with sf.SoundFile(out_wav_path, mode='w', samplerate=sr, channels=1) as f_out:
                for i in range(n_chunks):
                    pct = (i + 1) / n_chunks
                    p_text.text(f"修复进度: {int(pct*100)}% | 处理中段 {i+1}/{n_chunks}")

                    s, e = i * chunk_len, min((i + 1) * chunk_len, waveform.shape[1])
                    chunk_data = waveform[:, s:e]

                    enhanced = enhance(model, df_state, chunk_data, atten_lim_db=actual_atten_lim)

                    if i == 0: # 截取开头30秒用于图谱预览，避免画长图刷爆内存
                        plot_orig = chunk_data.squeeze().numpy()[:30*sr]
                        plot_enh = enhanced.squeeze().cpu().numpy()[:30*sr]

                    f_out.write(enhanced.squeeze().cpu().numpy())
                    p_bar.progress(pct)
        except Exception as e:
            st.error(f"❌ 推理失败：{type(e).__name__}: {e}")
            paths.cleanup()
            st.stop()

        p_text.empty()
        p_bar.empty()

        st.success(f"完成！总耗时: {time.time() - start_t:.1f}s")

        # 4. Result Section
        st.markdown("### 修复结果")
        st.audio(out_wav_path)

        dl_col1, dl_col2 = st.columns(2)

        if is_video:
            with st.spinner("视频轨道重构中..."):
                out_mov_path = os.path.join(tempfile.gettempdir(), f"fixed_{paths.file_id}.mp4")
                try:
                    ac.reattach_audio_to_video(input_path, out_wav_path, out_mov_path)
                    with open(out_mov_path, "rb") as f:
                        dl_col1.download_button("下载修复后的视频", f, file_name=f"fixed_{uploaded_file.name}.mp4")
                except ac.FFmpegFailedError as e:
                    st.error(f"❌ 视频封装失败：{e}")
                finally:
                    # 视频封装完成后清理（音频文件保留到 session 结束）
                    if os.path.exists(out_mov_path):
                        try:
                            os.remove(out_mov_path)
                        except OSError:
                            pass
        else:
            with open(out_wav_path, "rb") as f:
                dl_col1.download_button("下载高质量 WAV", f, file_name=f"fixed_{uploaded_file.name}.wav")

        # 5. Insight (Optional)
        with st.expander("对比分析图谱 (预览提取的前30秒片段)"):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            plt.subplots_adjust(hspace=0.4)

            def draw(y, s_rate, ax, title):
                # 对片段进行轻量级降采样以便绘图，既快又安全
                if s_rate > 16000:
                    y = librosa.resample(y, orig_sr=s_rate, target_sr=16000)
                    s_rate = 16000
                S = librosa.feature.melspectrogram(y=y, sr=s_rate, n_mels=128)
                librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', sr=s_rate, ax=ax)
                ax.set_title(title, color='#0F172A', fontweight='bold')

            draw(plot_orig, sr, ax1, "原始样本 (前30秒片段)")
            draw(plot_enh, sr, ax2, "修复样本 (前30秒片段)")
            fig.patch.set_facecolor('#FFFFFF')
            for ax in [ax1, ax2]:
                ax.set_facecolor('#F8FAFC')
                ax.tick_params(colors='#475569')
            st.pyplot(fig)

        # 6. 清理临时文件（音频原始+工作文件；输出 wav 留着让用户多下载几次）
        try:
            paths.input_path.unlink(missing_ok=True)
            paths.work_wav_path.unlink(missing_ok=True)
        except OSError:
            pass
else:
    st.info("请在上方框内上传音频/视频文件开始修复。")

st.caption("Engine: DeepFilterNet3 | Architecture: Apple Silicon 原生支持")