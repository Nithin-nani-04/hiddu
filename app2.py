import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
import os

# --- User management files ---
USER_DATA_FILE = 'users.json'

# Load user data
def load_users():
    if not os.path.exists(USER_DATA_FILE):
        return {}
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

# Save user data
def save_users(users):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f)

# Add new user
def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = {'password': password, 'uploads': []}
    save_users(users)
    return True

# Authenticate user
def authenticate_user(username, password):
    users = load_users()
    return username in users and users[username]['password'] == password

# Save uploaded image info for user
def save_upload(username, image_name):
    users = load_users()
    if username in users:
        users[username]['uploads'].append(image_name)
        save_users(users)

# --- Load model ---
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 9)  # 9 classes
    model.load_state_dict(torch.load('pest_classifier_resnet50.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

class_names = ['aphids', 'yworm', 'beetle', 'llworm', 'hopper', 
               'mites', 'mosquito', 'sawfly', 'stem_borer']

# --- Pesticide Information ---
pesticide_info = {
    "aphids": {
        "info": "Aphids are small sap-sucking insects that damage plants by feeding on sap.",
        "pesticides": [
            {"name": "Imidacloprid", "image": "images/imidacloprid.png"},
            {"name": "Malathion", "image": "images/malathion.png"},
            {"name": "Neem oil", "image": "images/neem_oil.png"}
        ]
    },
    "yworm": {
        "info": "Yellow worm damages crops by feeding on leaves.",
        "pesticides": [
            {"name": "Chlorpyrifos", "image": "images/chlorpyrifos.png"},
            {"name": "Fipronil", "image": "images/fipronil.png"}
        ]
    },
    "beetle": {
        "info": "Beetles can damage crops by feeding on leaves, stems, and roots.",
        "pesticides": [
            {"name": "Carbaryl", "image": "images/carbaryl.png"},
            {"name": "Permethrin", "image": "images/permethrin.png"},
            {"name": "Deltamethrin", "image": "images/deltamethrin.png"}
        ]
    },
    "llworm": {
        "info": "Leaf-roller worms wrap leaves and feed inside the rolled leaf.",
        "pesticides": [
            {"name": "Bacillus thuringiensis", "image": "images/bt.png"},
            {"name": "Spinosad", "image": "images/spinosad.png"}
        ]
    },
    "hopper": {
        "info": "Hoppers damage plants by sucking sap and transmitting plant pathogens.",
        "pesticides": [
            {"name": "Thiamethoxam", "image": "images/thiamethoxam.png"},
            {"name": "Dinotefuran", "image": "images/dinotefuran.png"}
        ]
    },
    "mites": {
        "info": "Mites are tiny arthropods that cause damage by feeding on plant tissue.",
        "pesticides": [
            {"name": "Abamectin", "image": "images/abamectin.png"},
            {"name": "Fenpyroximate", "image": "images/fenpyroximate.png"},
            {"name": "Propargite", "image": "images/propargite.png"}
        ]
    },
    "mosquito": {
        "info": "Mosquitoes are known for spreading diseases but can also damage crops indirectly.",
        "pesticides": [
            {"name": "Pyrethroids", "image": "images/pyrethroids.png"},
            {"name": "Larvicides", "image": "images/larvicides.png"}
        ]
    },
    "sawfly": {
        "info": "Sawflies are wasp-like insects whose larvae feed on plant leaves.",
        "pesticides": [
            {"name": "Spinosad", "image": "images/spinosad.png"},
            {"name": "Carbaryl", "image": "images/carbaryl.png"}
        ]
    },
    "stem_borer": {
        "info": "Stem borers feed on the stems and can cause significant yield loss.",
        "pesticides": [
            {"name": "Chlorantraniliprole", "image": "images/chlorantraniliprole.png"},
            {"name": "Flubendiamide", "image": "images/flubendiamide.png"},
            {"name": "Cypermethrin", "image": "images/cypermethrin.png"}
        ]
    }
}


# --- Image transform ---
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- Prediction function ---
def predict(image):
    tensor = transform_image(image)
    outputs = model(tensor)
    _, preds = torch.max(outputs, 1)
    return class_names[preds.item()]

# --- Streamlit UI ---
st.set_page_config(page_title="Pest Classifier with Authentication")

# Session state for login info
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.success("Logged out successfully.")

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome {username}!")
        else:
            st.error("Invalid username or password.")

def register_page():
    st.title("Register")
    username = st.text_input("Choose Username")
    password = st.text_input("Choose Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if password != confirm_password:
            st.error("Passwords do not match.")
        elif register_user(username, password):
            st.success("User registered successfully! Please login.")
        else:
            st.error("Username already exists.")

def main_app():
    st.sidebar.title(f"Hello, {st.session_state.username}")
    if st.sidebar.button("Logout"):
        logout()
        return

    page = st.sidebar.selectbox("Navigation", ["Home", "Pesticide Info", "My Uploads", "About"])

    if page == "Home":
        st.title("Pest Detection and Pesticide Recommendation")
        uploaded_file = st.file_uploader("Upload Pest Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            pest_class = predict(image)
            st.success(f"Predicted Pest: **{pest_class}**")
            save_upload(st.session_state.username, uploaded_file.name)

            info = pesticide_info.get(pest_class, {})
            if info:
                st.write(info["info"])
                st.subheader("Recommended Pesticides")
                for pesticide in info["pesticides"]:
                    image_path = pesticide["image"]
                    if os.path.exists(image_path):
                        img = Image.open(image_path)
                        st.image(img, caption=pesticide["name"], width=100)
                    else:
                        st.error(f"Image not found: {image_path}")

    elif page == "Pesticide Info":
        st.title("Pesticide Information")
        pest = st.selectbox("Select Pest", class_names)
        info = pesticide_info.get(pest, {})
        if info:
            st.write(info["info"])
            for pesticide in info["pesticides"]:
                img = Image.open(pesticide["image"])
                st.image(img, caption=pesticide["name"], width=100)

    elif page == "My Uploads":
        st.title("My Uploaded Images")
        uploads = load_users().get(st.session_state.username, {}).get('uploads', [])
        st.write(uploads if uploads else "No uploads yet.")

    else:
        st.title("About This App")
        st.markdown("This app classifies crop pests and recommends pesticides using ResNet50 and Streamlit.")

# Run the app
if not st.session_state.logged_in:
    auth_choice = st.sidebar.selectbox("Select Option", ["Login", "Register"])
    if auth_choice == "Login":
        login_page()
    else:
        register_page()
else:
    main_app()
