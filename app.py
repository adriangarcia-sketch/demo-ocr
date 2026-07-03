import streamlit as st
import os
import json
import datetime
from google import genai
from google.genai import types
from google.cloud import vision, storage, bigquery, documentai

# Inicializar clientes con Vertex AI activado explícitamente
storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()
docai_client = documentai.DocumentProcessorServiceClient()
ai_client = genai.Client(vertexai=True)
bq_client = bigquery.Client()

BUCKET_NAME = "demo-comparativa-ine-xertica-presales-data-service"
PROJECT_ID = "xertica-presales-data-service"

# Credenciales de tu procesador Custom de Document AI
DOCAI_PROJECT_ID = "843527592623"
DOCAI_LOCATION = "us"
DOCAI_PROCESSOR_ID = "204606654c72b17a"

st.set_page_config(layout="wide")
st.title("🛡️ Centro de Procesamiento de Identificaciones Agnóstico")
st.subheader("Análisis en Paralelo: OCR Tradicional vs Document AI Custom vs IA Multimodal (Gemini)")

uploaded_file = st.file_uploader("Sube una identificación (INE, Pasaporte, Visa americana, etc.)", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    with st.spinner("Procesando documento simultáneamente en todos los motores..."):
        temp_path = uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # 1. Almacenamiento seguro en el Bucket creado
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(temp_path)
        blob.upload_from_filename(temp_path)
        gcs_uri = f"gs://{BUCKET_NAME}/{temp_path}"
        
        # ----------------------------------------------------------------------
        # MOTOR 1: CLOUD VISION (OCR Base)
        # ----------------------------------------------------------------------
        image = vision.Image()
        image.source.image_uri = gcs_uri
        response_vision = vision_client.text_detection(image=image)
        texto_ocr = response_vision.text_annotations[0].description if response_vision.text_annotations else "No se extrajo texto plano."
        
        # ----------------------------------------------------------------------
        # MOTOR 2: DOCUMENT AI (Tu procesador especializado)
        # ----------------------------------------------------------------------
        with open(temp_path, "rb") as image_file:
            image_content = image_file.read()
            
        name = f"projects/{DOCAI_PROJECT_ID}/locations/{DOCAI_LOCATION}/processors/{DOCAI_PROCESSOR_ID}"
        raw_document = documentai.RawDocument(content=image_content, mime_type=uploaded_file.type)
        request_docai = documentai.ProcessRequest(name=name, raw_document=raw_document)
        
        entidades_docai = {}
        try:
            result_docai = docai_client.process_document(request=request_docai)
            for entity in result_docai.document.entities:
                entidades_docai[entity.type_] = entity.mention_text
        except Exception as e:
            entidades_docai = {"status": "Error en procesador Custom", "detalle": str(e)}

        # ----------------------------------------------------------------------
        # MOTOR 3: GEMINI 2.5 FLASH (Cognitivo Humano No-JSON)
        # ----------------------------------------------------------------------
        prompt = """
        Identifica el tipo de documento adjunto (INE, Pasaporte, Visa u otro).
        Extrae todos los campos clave relevantes disponibles y ordénalos en formato de viñetas Markdown simples:
        
        **Tipo de Identificación:** [Tipo]
        **[Nombre del Campo]:** [Valor]
        
        Sé limpio en la respuesta, no pongas introducciones ni bloques de código formateados (```). Go straight to the fields.
        """
        response_gemini = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[types.Part.from_uri(file_uri=gcs_uri, mime_type=uploaded_file.type), prompt]
        )
        analisis_gemini = response_gemini.text

        # ----------------------------------------------------------------------
        # INSERCIÓN EN BIGQUERY (Estructura Agnóstica Flexible)
        # ----------------------------------------------------------------------
        rows = [{
            "texto_crudo_vision": texto_ocr,
            "datos_estructurados_document_ai": entidades_docai, # Se guarda como JSON nativo de BQ
            "analisis_cognitivo_gemini": analisis_gemini,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }]
        bq_client.insert_rows_json(f"{PROJECT_ID}.demo_dataset.comparativa_ocr_gemini", rows)

        # ----------------------------------------------------------------------
        # RENDERIZADO VISUAL EN STREAMLIT (Tres Columnas)
        # ----------------------------------------------------------------------
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.error("❌ 1. OCR Tradicional (Texto Plano)")
            st.text_area("Resultado Crudo (Vision API):", value=texto_ocr, height=400, key="ocr_box")
            
        with col2:
            st.warning("⚙️ 2. Document AI (Procesador Custom)")
            if entidades_docai:
                for k, v in entidades_docai.items():
                    st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
            else:
                st.info("Procesado con éxito pero no se mapearon entidades específicas.")
            
        with col3:
            st.success("✨ 3. IA Generativa Cognitiva (Gemini)")
            st.markdown(analisis_gemini)
            
        st.info("💾 Éxito: Auditoría registrada en BigQuery de forma agnóstica sin importar el tipo de ID.")
        os.remove(temp_path)
