"""
Medical Image Analysis Web App - Hugging Face Spaces Version
Streamlit application that performs:
1. Cell segmentation using HoVerNet + U-Net
2. Caption generation using RGB + Segmentation (4-channel input)

Input: PNG/JPG pathology image
Output: Segmentation visualization + Medical caption
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pickle
import nltk
from torchvision import transforms
import timm
import io
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import sys
import torch.nn.functional as F
from scipy.ndimage import center_of_mass

# Add paths for segmentation 
sys.path.append('./pathSegmentation')
sys.path.append('./utils')

from pathSeg.ml.hovernet import HoVerNet, post_process_batch_hovernet
import segmentation_models_pytorch as smp
from torchvision.transforms import ToTensor

# Download NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt', quiet=True)
    
# try:
#     nltk.data.find('tokenizers/punkt_tab')
# except LookupError:
#     nltk.download('punkt_tab', quiet=True)

import os
nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
    nltk.download('punkt', quiet=True)
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt_tab')):
    nltk.download('punkt_tab', quiet=True)


# Page config
st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .seg-box {
        background-color: #f0fff0;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .step-indicator {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
            
            
    [data-testid="stSidebar"] {
        height: 100vh !important;
        overflow-y: hidden !important;
    }
    
    [data-testid="stSidebarContent"] {
        height: 100vh !important;
        overflow-y: scroll !important;
    }

            


                
        /* targets columns nested inside the 'image-section-fence' */
    .image-section-fence [data-testid="column"] {
        aspect-ratio: 1 / 1 !important;        /* Keeps image area square */
        display: flex !important;              /* Enables alignment control */
        flex-direction: column !important;     /* Stacks image and caption vertically */
        min-height: 250px !important;          /* Prevents collapse if width calculation lags */
    }

    /* Ensures Streamlit image wrapper fills the square */
    .image-section-fence [data-testid="stImage"] {
        height: 100% !important;
        width: 100% !important;
    }

    /* Ensures actual image pixels fill the wrapper without distortion */
    .image-section-fence [data-testid="stImage"] img {
        object-fit: contain !important;
        height: 100% !important;
        width: 100% !important;
    }

                
    /* Prevent auto-scroll when fragment reruns */
    [data-testid="stSidebar"] input[type="checkbox"]:focus {
        scroll-margin: 0 !important;
    }

    [data-testid="stVerticalBlock"] {
        scroll-margin-top: 0 !important;
    }

    /* Stabilize fragment container specifically */
    .image-section-fence {
        scroll-margin-top: 0 !important;
        scroll-margin-bottom: 0 !important;
    }


</style>
""", unsafe_allow_html=True)

# Configuration
IMG_SIZE = 512
EMBED_SIZE = 300
HIDDEN_SIZE = 512
NUM_LAYERS = 4
NUM_HEADS = 15
IMAGE_FEATURE_DIM = 1280

# Segmentation class configuration
CLASS_NAMES = ['Stroma', 'Immune', 'Normal/Epithelial', 'Tumor']
COLORS = {
    0: [1, 0.8, 0, 0.7],      # Yellow - Stroma
    1: [0, 1, 0, 0.7],         # Green - Immune
    2: [0, 0, 1, 0.7],         # Blue - Normal
    3: [1, 0, 0, 0.7],         # Red - Tumor
}


# ============================================================================
# SEGMENTATION FUNCTIONS
# ============================================================================

def post_process_unet(outputs):
    """Post-process U-Net output"""
    predict = F.softmax(outputs, dim=1)
    predict = predict.cpu().detach().numpy() > 0.5
    
    predict_temp = np.zeros((512, 512))
    for c in range(4):
        predict_temp = np.where(predict[0, c] == 1, c, predict_temp)
    
    return predict_temp


def combine_predictions(cell_seg, tissue_class):
    """Combine cell and tissue segmentation"""
    cell_total = np.zeros((512, 512, 4))
    
    for cell_id in range(1, int(np.max(cell_seg)) + 1):
        cell_mask = (cell_seg == cell_id).astype(int)
        center = center_of_mass(cell_mask)
        
        if np.isnan(center[0]) or np.isnan(center[1]):
            continue
        
        y, x = int(center[0]), int(center[1])
        tissue_type = int(tissue_class[y, x])
        cell_total[:, :, tissue_type] += cell_mask * cell_id
    
    return cell_total


def compress_segmentation_to_single_channel(segmentation):
    """Compress 4-channel segmentation to single channel"""
    seg_1ch = np.zeros((512, 512), dtype=np.uint8)
    
    for i in range(4):
        mask = segmentation[:, :, i] > 0
        seg_1ch[mask] = i + 1
    
    return seg_1ch


def create_colored_overlay(segmentation, colors):
    """Create colored visualization of segmentation"""
    colored = np.zeros((512, 512, 4))
    
    for channel in range(4):
        mask = segmentation[:, :, channel] > 0
        for i in range(4):
            colored[:, :, i] = np.where(mask, colors[channel][i], colored[:, :, i])
    
    return colored


def preprocess_for_segmentation(image):
    """Preprocess image for segmentation"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if image.size != (512, 512):
        image = image.resize((512, 512), Image.BILINEAR)
    
    transform = ToTensor()
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor


def run_segmentation(image, model1, model2, device):
    """Run segmentation inference on uploaded image"""
    image_tensor = preprocess_for_segmentation(image).float().to(device)
    
    with torch.no_grad():
        outputs1 = model1(image_tensor)
        cell_seg = post_process_batch_hovernet(outputs1, n_classes=None)[0]
        
        outputs2 = model2(image_tensor)
        tissue_class = post_process_unet(outputs2)
    
    final_segmentation = combine_predictions(cell_seg, tissue_class)
    
    return final_segmentation, cell_seg, tissue_class


def create_overlay_image(original_image, segmentation, show_mask=True, alpha=0.5):
    """Create overlay image as numpy array"""
    img_array = np.array(original_image.resize((512, 512)))
    
    if not show_mask:
        return img_array, img_array.copy()
    
    colored_mask = create_colored_overlay(segmentation, COLORS)
    colored_mask_rgb = (colored_mask[:, :, :3] * 255).astype(np.uint8)
    
    overlay_array = img_array.copy().astype(np.float32)
    
    for i in range(3):
        mask_channel = colored_mask[:, :, 3] > 0
        overlay_array[:, :, i] = np.where(
            mask_channel,
            img_array[:, :, i] * (1 - alpha) + colored_mask_rgb[:, :, i] * alpha,
            img_array[:, :, i]
        )
    
    overlay_array = overlay_array.astype(np.uint8)
    
    return img_array, overlay_array


def calculate_segmentation_statistics(segmentation):
    """Calculate detailed segmentation statistics"""
    stats = {}
    
    for i, class_name in enumerate(CLASS_NAMES):
        channel = segmentation[:, :, i]
        num_regions = len(np.unique(channel)) - 1
        num_pixels = np.count_nonzero(channel)
        coverage = (num_pixels / (512 * 512)) * 100
        
        stats[class_name] = {
            'regions': num_regions,
            'pixels': num_pixels,
            'coverage': coverage
        }
    
    return stats


# ============================================================================
# CAPTION GENERATION MODELS
# ============================================================================

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx.get('<unk>', 0)
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class FeatureExtractor(nn.Module):
    """Feature extractor using EfficientNetV2-S with 4-channel input"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.backbone = timm.create_model(
            'efficientnetv2_s',
            pretrained=False,
            in_chans=4,
            num_classes=0,
            global_pool='avg'
        )

    def forward(self, x):
        return self.backbone(x)


class AttentionMILModel(nn.Module):
    """Attention-based Multiple Instance Learning model"""
    def __init__(self, num_classes, image_feature_dim, feature_extractor):
        super(AttentionMILModel, self).__init__()
        self.num_classes = num_classes
        self.image_feature_dim = image_feature_dim
        self.feature_extractor = feature_extractor
        
        self.attention = nn.Sequential(
            nn.Linear(image_feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.classification_layer = nn.Linear(image_feature_dim, num_classes)

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.size()
        features = self.feature_extractor(inputs)
        features = features.view(batch_size, -1)
        logits = self.classification_layer(features)
        return logits


class DecoderTransformer(nn.Module):
    """Transformer-based decoder for caption generation."""
    def __init__(self, embed_size, vocab_size, num_heads, hidden_size, num_layers, max_seq_length=100):
        super(DecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_size))
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def sample(self, features, max_seq_length=None):
        """Generate captions using greedy search with proper stopping."""
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        
        batch_size = features.size(0)
        sampled_ids = []
        
        input_caption = torch.ones(batch_size, 1).long().to(features.device)
        memory = features.unsqueeze(0)
        
        has_ended = torch.zeros(batch_size, dtype=torch.bool).to(features.device)
        
        for i in range(max_seq_length):
            input_embedded = self.embed(input_caption) + \
                           self.positional_encoding[:, :input_caption.size(1), :]
            input_embedded = input_embedded.permute(1, 0, 2)
            tgt_mask = self.generate_square_subsequent_mask(input_embedded.size(0)).to(features.device)
            transformer_output = self.transformer_decoder(input_embedded, memory, tgt_mask=tgt_mask)
            transformer_output = transformer_output.permute(1, 0, 2)
            output = self.linear(transformer_output[:, -1, :])
            _, predicted = output.max(1)
            
            is_end = (predicted == 2)
            has_ended = has_ended | is_end
            
            if has_ended.all():
                break
            
            sampled_ids.append(predicted)
            input_caption = torch.cat([input_caption, predicted.unsqueeze(1)], dim=1)
        
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def idx2word(vocab, indices):
    """Convert indices to words."""
    sentence = []
    aa = indices.cpu().numpy()
    for index in aa:
        word = vocab.idx2word[index]
        sentence.append(word)
    return sentence


def word2sentence(words_list):
    """Convert word list to sentence string."""
    sentence = ''
    for word in words_list:
        if word.isalnum():
            sentence += ' ' + word
        else:
            sentence += word
    return sentence.strip()


def preprocess_for_caption(image, segmentation, device):
    """Preprocess image and segmentation for caption generation"""
    transform_rgb = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    rgb_tensor = transform_rgb(image)
    
    seg_1ch = compress_segmentation_to_single_channel(segmentation)
    
    seg_tensor = torch.from_numpy(seg_1ch.astype('float32')) / 4.0
    seg_tensor = seg_tensor.unsqueeze(0)
    
    combined_tensor = torch.cat([rgb_tensor, seg_tensor], dim=0)
    
    combined_tensor = torch.nan_to_num(combined_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    
    combined_tensor = combined_tensor.unsqueeze(0)
    
    return combined_tensor.to(device)


def generate_caption(image, segmentation, vocab, feature_extractor, projection_layer, caption_decoder, device):
    """Generate caption for an image using RGB + Segmentation"""
    input_tensor = preprocess_for_caption(image, segmentation, device)
    
    with torch.no_grad():
        features = feature_extractor(input_tensor)
        features = projection_layer(features)
        sampled_ids = caption_decoder.sample(features)
    
    predicted_caption = idx2word(vocab, sampled_ids[0])
    
    clean_caption = []
    for word in predicted_caption:
        if word in ['<end>', '<pad>']:
            break
        if word not in ['<start>', '<unk>']:
            clean_caption.append(word)
    
    caption_text = word2sentence(clean_caption)
    
    return caption_text, clean_caption


@st.cache_resource
def load_all_models(seg_model1_path, seg_model2_path, vocab_path, encoder_path, projection_path, decoder_path):
    """Load all models"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load segmentation models
    seg_model1 = HoVerNet(n_classes=None)
    seg_model1.load_state_dict(torch.load(seg_model1_path, map_location=device))
    seg_model1.to(device)
    seg_model1.eval()
    
    seg_model2 = smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4,
    )
    seg_model2.load_state_dict(torch.load(seg_model2_path, map_location=device))
    seg_model2.to(device)
    seg_model2.eval()
    
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Load encoder
    checkpoint = torch.load(encoder_path, map_location=device)
    
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor_state = {k: v for k, v in checkpoint.items() if k.startswith('backbone.')}
    feature_extractor.load_state_dict(feature_extractor_state, strict=False)
    feature_extractor.eval()
    
    # Projection layer
    projection_layer = nn.Linear(IMAGE_FEATURE_DIM, EMBED_SIZE).to(device)
    
    if any(k.startswith('proj.') for k in checkpoint.keys()):
        found_projection = False
        for i in range(10):
            weight_key = f'proj.{i}.weight'
            bias_key = f'proj.{i}.bias'
            
            if weight_key in checkpoint:
                weight = checkpoint[weight_key]
                if weight.shape == torch.Size([EMBED_SIZE, IMAGE_FEATURE_DIM]):
                    projection_layer.weight.data = weight
                    projection_layer.bias.data = checkpoint[bias_key]
                    found_projection = True
                    break
        
        if not found_projection:
            if Path(projection_path).exists():
                projection_layer.load_state_dict(torch.load(projection_path, map_location=device))
    else:
        if Path(projection_path).exists():
            projection_layer.load_state_dict(torch.load(projection_path, map_location=device))
    
    projection_layer.eval()
    
    # Decoder
    caption_decoder = DecoderTransformer(
        EMBED_SIZE, 
        len(vocab), 
        NUM_HEADS, 
        HIDDEN_SIZE, 
        NUM_LAYERS
    ).to(device)
    caption_decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    caption_decoder.eval()
    
    return seg_model1, seg_model2, vocab, feature_extractor, projection_layer, caption_decoder, device
# ============================================================================
# FRAGMENT FOR VISUALIZATION (FIXED)
# ============================================================================
@st.fragment
def render_visualization_fragment(image, segmentation, show_mask_overlay):
    """
    Renders the image visualization first, 
    then places a stable transparency slider directly underneath.
    """
    st.markdown("---")
    st.subheader("🎨 Segmentation Visualization")
    
    # 1. Access the slider value directly from its 'key' in session_state.
    # If it's the first run, we provide a default of 0.5.
    current_alpha = st.session_state.get("mask_slider_key", 0.5)

    # # 2. Generate and Display the Images using the stable value
    # original_array, overlay_array = create_overlay_image(
    #     image, 
    #     segmentation, 
    #     show_mask=show_mask_overlay,
    #     alpha=current_alpha
    # )
    
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.image(original_array, caption='Original Image', width='stretch')
    
    # with col2:
    #     caption_img = f'Overlay (α={current_alpha:.1f})' if show_mask_overlay else 'Original'
    #     st.image(overlay_array, caption=caption_img, width='stretch')



    # 2. Pre-create containers first
    col1, col2 = st.columns(2)
    image_container_1 = col1.container()
    image_container_2 = col2.container()
    slider_container = st.container()
    legend_container = st.container()
    
    # 3. Generate images
    original_array, overlay_array = create_overlay_image(
        image, 
        segmentation, 
        show_mask=show_mask_overlay,
        alpha=current_alpha
    )
    
    # 4. Fill image containers
    with image_container_1:
        st.image(original_array, caption='Original Image')
    
    with image_container_2:
        caption_img = f'Overlay (α={current_alpha:.1f})' if show_mask_overlay else 'Original'
        st.image(overlay_array, caption=caption_img)

    # 3. Place the Slider UNDER the images

    # We don't manually assign a 'value=' here from a variable that changes.
    # Instead, we let the 'key' manage the state internally.
    with slider_container:
        st.slider(
            "Adjust Mask Transparency", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            step=0.1,
            key="mask_slider_key",
            disabled=not show_mask_overlay # Streamlit handles the state sync automatically now
        )
    
    # 4. Legend at the bottom
    with legend_container:
        st.markdown("""
        <div style="text-align: center; margin-top: 1rem;">
            <span style="background-color: rgba(255, 204, 0, 0.7); padding: 5px 15px; margin: 5px; border-radius: 5px;">🟡 Stroma</span>
            <span style="background-color: rgba(0, 255, 0, 0.7); padding: 5px 15px; margin: 5px; border-radius: 5px;">🟢 Immune</span>
            <span style="background-color: rgba(0, 0, 255, 0.7); padding: 5px 15px; margin: 5px; border-radius: 5px; color: white;">🔵 Normal</span>
            <span style="background-color: rgba(255, 0, 0, 0.7); padding: 5px 15px; margin: 5px; border-radius: 5px; color: white;">🔴 Tumor</span>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# MAIN APP (COMPLETE 1:1 REPLACEMENT)
# ============================================================================

def main():
    import pandas as pd

    # Hardcoded paths
    seg_model1_path = "model/best_seg_BR_cell.pt"
    seg_model2_path = "model/best_seg_BR_class.pt"
    vocab_path = "vocab.pkl"
    encoder_path = "encoder_4ch.pth"
    projection_path = "projection.pth"
    decoder_path = "decoder.pth"

    # Initialize session state
    if 'segmentation_result' not in st.session_state:
        st.session_state.segmentation_result = None
    if 'caption_result' not in st.session_state:
        st.session_state.caption_result = None
    if 'stats_result' not in st.session_state:
        st.session_state.stats_result = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    
    # Custom CSS and Header (Same as before)
    # ... (Your CSS block goes here)

    # Sidebar
    with st.sidebar:

        st.header("⚙️ Configuration")
        st.markdown("---")
        
        st.subheader("Processing Options")
        show_detailed_vis = st.checkbox("Show Detailed Visualization", value=True)
        show_statistics = st.checkbox("Show Statistics", value=True)
        
        st.subheader("Visualization Options")
        show_mask_overlay = st.checkbox("Show Segmentation Mask", value=True)
        
        # ℹ️ Re-inserted: About Section
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.info("""
        **Model:** RGB + Segmentation
        
        **Caption:** 4-channel input
        - EfficientNetV2-S encoder
        - Transformer decoder
        """)
        
        # 🖥️ Re-inserted: System Info Section
        st.markdown("---")
        st.subheader("🖥️ System Info")
        device_type = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.write(f"**Device:** {device_type}")
        if torch.cuda.is_available():
            st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")

        st.markdown("---")
        st.markdown("<h3 style='font-size: 1.15rem;'>Streamlit page made by <br> Son, Hyuk Jae</h3>", unsafe_allow_html=True)

        
    # Model loading logic
    with st.spinner("🔄 Loading models..."):
        try:
            seg_model1, seg_model2, vocab, feature_extractor, projection_layer, caption_decoder, device = load_all_models(
                seg_model1_path, seg_model2_path, vocab_path, encoder_path, projection_path, decoder_path
            )
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return

    # File upload
    st.markdown("---")
    st.header("📤 Upload Pathology Image")
    uploaded_file = st.file_uploader("Choose image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is None:
        st.markdown("""
                    <div style="text-align: center; margin-top: 0.5rem; font-size: 0.9rem;">
    📥 Google drive link for test images: <a href="https://drive.google.com/drive/folders/1Fw4d6-QMPjcXvQkAPhJYpePAl5X9jHzZ?usp=drive_link" target="_blank">Download here</a>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display Image Info
        st.subheader("📊 Image Information")
        col1, col2, col3 = st.columns(3)
        col1.metric("Width", f"{image.size[0]} px")
        col2.metric("Height", f"{image.size[1]} px")
        col3.metric("Format", image.format or "Unknown")

        if st.button("🚀 Run Complete Analysis", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                status_text.markdown('<div class="step-indicator">Step 1/2: Segmenting...</div>', unsafe_allow_html=True)
                segmentation, _, _ = run_segmentation(image, seg_model1, seg_model2, device)
                st.session_state.segmentation_result = segmentation
                st.session_state.uploaded_image = image
                
                status_text.markdown('<div class="step-indicator">Step 2/2: Captioning...</div>', unsafe_allow_html=True)
                caption_text, caption_words = generate_caption(image, segmentation, vocab, feature_extractor, projection_layer, caption_decoder, device)
                st.session_state.caption_result = (caption_text, caption_words)
                
                st.session_state.stats_result = calculate_segmentation_statistics(segmentation)
                progress_bar.progress(100)
                st.success("✅ Complete!")
            except Exception as e:
                st.error(f"Error: {e}")

        # --- RESULTS DISPLAY ---
        if st.session_state.segmentation_result is not None:
            # 1. Stats
            if show_statistics:
                st.markdown('<div class="seg-box"><h2> Segmentation Results</h2></div>', unsafe_allow_html=True)
                cols = st.columns(4)
                color_hex = ['#FFD700', '#00FF00', '#0000FF', '#FF0000']
                for i, (name, s) in enumerate(st.session_state.stats_result.items()):
                    with cols[i]:
                        st.markdown(f'<div class="metric-card"><h3 style="color:{color_hex[i]}">{name}</h3>'
                                    f'<b>{s["regions"]}</b> regions<br>{s["coverage"]:.2f}% cov</div>', unsafe_allow_html=True)

            # 2. Fragment Visualization (THE FIX)
            if show_detailed_vis:
                st.markdown('<div class="image-section-fence">', unsafe_allow_html=True)
                render_visualization_fragment(st.session_state.uploaded_image, st.session_state.segmentation_result, show_mask_overlay)
                st.markdown('</div>', unsafe_allow_html=True)

            # 3. Caption
            if st.session_state.caption_result:
                caption_text, caption_words = st.session_state.caption_result
                st.markdown(f'<div class="result-box"><h2>📝 AI Caption</h2><p style="font-size:1.4rem">{caption_text}</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()