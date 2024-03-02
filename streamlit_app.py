############################################################################################
############################################################################################
# App de interactuar con una URL (link de sitio web)
############################################################################################
############################################################################################

# https://www.youtube.com/watch?v=bupx08ZgSFg&t=4059s&ab_channel=AlejandroAO-Software%26Ai
# https://github.com/alejandro-ao/chat-with-websites/blob/master/src/app.py

# https://platform.openai.com/account/api-keys
# https://openai.com/pricing



#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [A] Importacion de librerias
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# !pip install --upgrade openai --user
# !pip install --upgrade langchain --user
# !pip install langchain_openai --user
# !pip install --upgrade streamlit --user
# !pip show openai

# Obtener versiones de paquetes instalados
# !pip list > requirements.txt

import streamlit as st

from openai import OpenAI

from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI # from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import time

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [B] Creacion de funciones internas utiles
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def Cadena_Modelo(
  st_Texto_links, # un string de listado de links separados por coma
  st_ApiKey # una api key de openAI ingresada
):
  
  # pasar texto ingresado a lista 
  Lista_links = st_Texto_links.split(',')  
  
  # cargar documento desde sitio web
  documento = WebBaseLoader(Lista_links).load()
  
  # crear divisor de texto
  divisor_texto = RecursiveCharacterTextSplitter( # CharacterTextSplitter
    chunk_size=1000, # tamaño de cada bloque de extraccion
    chunk_overlap=50, # traslape de cada bloque
    separators=[' ',',','\n'] # separadores para eventualmente forzar separacion 
    )

  # aplicar divisor de texto
  documentos = divisor_texto.split_documents(documento)
    
  # crear base vectorial (puede ser tambien con Chroma)
  base_vectorial = FAISS.from_documents(
    documents = documentos,
    embedding = OpenAIEmbeddings(openai_api_key = st_ApiKey)
    )
    
  # crear modelo (puede ser importado desde distintas fuentes)
  Modelo_LLM = ChatOpenAI(
    openai_api_key = st_ApiKey,
    model_name='gpt-3.5-turbo' # parametro opcional para definir modelo a usar
    )

  # crear objeto retriever 
  Base_vectorial_retriever = base_vectorial.as_retriever()

  # crear objeto de consulta
  Consulta_formato_usuario = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name='historial_chat'),
    ('user','{input}'),
    ('user','Dada la conversacion anterior, genera una consulta de busqueda que obtenga informacion relevante para la conversacion')
    ])

  # crear cadena de consulta
  Cadena_retriever = create_history_aware_retriever(
    Modelo_LLM, 
    Base_vectorial_retriever, 
    Consulta_formato_usuario
    )
    
  # crear objeto de consulta2
  Consulta_formato_sistema = ChatPromptTemplate.from_messages([
    ('system','Responde la consulta del usuario basado en el siguiente contexto:\n\n{context}'),
    MessagesPlaceholder(variable_name='historial_chat'),
    ('user','{input}'),
  ])


  # crear cadena de stuff_documents
  Cadena_stuff_documents = create_stuff_documents_chain(
    Modelo_LLM,
    Consulta_formato_sistema
    )

  # crear cadena rag
  Cadena_rag = create_retrieval_chain(
    Cadena_retriever, 
    Cadena_stuff_documents
    )

  
  # arrojar entregable 
  return Cadena_rag
  


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [C] Generacion de la App
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

st.set_page_config(layout='wide')

# titulo inicial 
st.markdown('## :globe_with_meridians: Interactuar con Sitios Web :globe_with_meridians:')

# autoria 
st.sidebar.markdown('**Autor :point_right: [Sebastian Barrera](https://www.linkedin.com/in/sebasti%C3%A1n-nicolas-barrera-varas-70699a28)**')

# ingresar OpenAI api key
usuario_api_key = st.sidebar.text_input(
  label='Tu OpenAI API key :key:',
  placeholder='Pega aca tu openAI API key',
  type='password'
  )

# ingresar listado de links separados por coma
texto_links = st.sidebar.text_area(
  'Ingresa aca los links separados por ","',
  )

# colocar separador para mostrar en sidebar otras cosas
st.sidebar.markdown('---')

# colocar si se quiere generar audio
generar_audio = st.sidebar.toggle('Generar Audio?')


#_____________________________________________________________________________
# comenzar a desplegar app una vez ingresado el archivo

if len(usuario_api_key)>0 and len(texto_links)>0:  
   
  #_____________________________________________________________________________
  # Aplicar todo lo relacionado al modelo de LLM para tener listas las consultas

  # creamos objeto cadena (usando funcion en cache creada antes)
  Cadena_LLM = Cadena_Modelo(
    st_Texto_links = texto_links,
    st_ApiKey = usuario_api_key
    )

  # definimos lista de historial chat
  historial_chat = [
    AIMessage(content='Hola, soy un ChatBot, dime como te puedo ayudar?')
    ]
    
  
  if 'messages' not in st.session_state:
    st.session_state.messages = []

  for message in st.session_state.messages:
    avatar_usar = '😎' if message['role']=='user' else '🤖'
    with st.chat_message(message['role'],avatar=avatar_usar):
      st.markdown(message['content'])

  if consulta := st.chat_input('Ingresa tu consulta aqui'):
    st.session_state.messages.append({'role': 'user', 'content': consulta})
    
    with st.chat_message('user',avatar='😎'):
      st.markdown(consulta)
      
    with st.chat_message('assistant',avatar='🤖'):
      
      # calcular respuesta de llm
      entregable_llm = Cadena_LLM.invoke({
        'historial_chat': historial_chat,
        'input': consulta
        })
      
      respuesta = entregable_llm['answer']
      
      # guardar respuesta en historial
      historial_chat.append(HumanMessage(content=consulta))
      historial_chat.append(AIMessage(content=respuesta))
      

      # mostrar parte por parte (solo estetico, dado que puede mostrar mas rapidamente la respuesta de inmediato)
      mensaje_entregable = st.empty()
      respuesta_total = ''

      for r in respuesta.split(' '):        
        respuesta_total += r+' '
        mensaje_entregable.markdown(respuesta_total + '▌')
        time.sleep(0.07)

      mensaje_entregable.markdown(respuesta_total)
    
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # mostrar audio de respuesta
    
    if generar_audio is True:
                
      # seleccionar tipo de voz que se utilizara
      tipos_voces = ['alloy','echo','fable','onyx','nova','shimmer']
      tipo_voz = st.sidebar.selectbox(
          'Que tipo de voz quieres?',
          tipos_voces
          )
    
      # crear cliente de openAI
      cliente_OpenAI = OpenAI(api_key = usuario_api_key)
      
      # usar modelo para pasar a audio
      respuesta_audio = cliente_OpenAI.audio.speech.create(
        model = 'tts-1',
        voice = tipo_voz,
        input = respuesta,
        speed = 1.15 # default 1.0 [0.25,4.0]
        )

      # guardar resultado
      respuesta_audio.stream_to_file('respuesta_audio.mp3')
          
      # mostrar audio de respuesta
      respuesta_audio2 = open('respuesta_audio.mp3', 'rb').read()
      st.sidebar.audio(respuesta_audio2, format='mp3')
    
    
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # mostrar referencias desde donde se obtiene la respuesta
        
    with st.sidebar.expander(':newspaper: Referencias',expanded=False):
      for c in entregable_llm['context']:
        st.markdown('---')
        st.markdown('##### Sitio: '+str(c.metadata['source']))
        st.write(c.page_content)
        st.write(' ')

           
    st.session_state.messages.append({'role': 'assistant', 'content': respuesta_total})
    
    


# !streamlit run App_LLM_URL.py

# para obtener TODOS los requerimientos de librerias que se usan
# !pip freeze > requirements.txt


# para obtener el archivo "requirements.txt" de los requerimientos puntuales de los .py
# !pipreqs "/Seba/Actividades Seba/Programacion Python/30_Streamlit App chat URL (01-03-24)/App/"

# Video tutorial para deployar una app en streamlitcloud
# https://www.youtube.com/watch?v=HKoOBiAaHGg&ab_channel=Streamlit
