import os
import json
import pickle
import numpy as np
import logging
import hashlib
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from functools import lru_cache
from app.config.config import EMBEDDING_INDEX_PATH
from langchain_community.vectorstores import FAISS

logger = logging.getLogger("query-engine")

class SemanticCache:
    def __init__(self, embedder, similarity_threshold: float = 0.8, max_cache_size: int = 100):
        self.embedder = embedder
        # Quando menor, mais tolerante
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.questions_path = os.path.join(EMBEDDING_INDEX_PATH, "questions")
        self._metadata_cache: Dict[str, OrderedDict] = {}
        self._ensure_directories()

    def _ensure_directories(self):
        """Garante que os diretórios necessários existam"""
        os.makedirs(self.questions_path, exist_ok=True)
        logger.info(f"Diretório de cache criado/verificado: {self.questions_path}")

    def _get_document_cache_path(self, document_id: str) -> str:
        """Retorna o caminho do cache para um documento específico"""
        return os.path.join(self.questions_path, document_id)

    def _get_query_hash(self, query: str) -> str:
        """Gera hash consistente para a query"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()

    def _get_faiss_path(self, document_id: str, query_hash: str) -> str:
        """Retorna o caminho do índice FAISS para uma query específica"""
        doc_path = self._get_document_cache_path(document_id)
        return os.path.join(doc_path, f"faiss_{query_hash}")

    def _get_metadata_path(self, document_id: str) -> str:
        """Retorna o caminho do arquivo de metadados"""
        doc_path = self._get_document_cache_path(document_id)
        return os.path.join(doc_path, "metadata.json")

    def _load_metadata(self, document_id: str) -> OrderedDict:
        """Carrega metadados do disco"""
        if document_id in self._metadata_cache:
            return self._metadata_cache[document_id]
        
        metadata_path = self._get_metadata_path(document_id)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    data = json.loads(text)
                    # Converter para OrderedDict mantendo ordem
                    metadata = OrderedDict()
                    for item in data.get('entries', []):
                        metadata[item['query_hash']] = {
                            'query': item['query'],
                            'response': item['response'],
                            'context_docs': item['context_docs'],
                            'timestamp': item.get('timestamp', 0)
                        }
                    self._metadata_cache[document_id] = metadata
                    logger.info(f"Metadados carregados para documento {document_id}: {len(metadata)} entradas")
                    return metadata
            except Exception as e:
                logger.error(f"Erro ao carregar metadados para {document_id}: {e}")
        
        # Criar novo se não existir
        self._metadata_cache[document_id] = OrderedDict()
        return self._metadata_cache[document_id]

    def _save_metadata(self, document_id: str):
        """Salva metadados no disco"""
        if document_id not in self._metadata_cache:
            return
        
        metadata_path = self._get_metadata_path(document_id)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        try:
            # Converter OrderedDict para formato JSON
            entries = []
            for query_hash, data in self._metadata_cache[document_id].items():
                entries.append({
                    'query_hash': query_hash,
                    'query': data['query'],
                    'response': data['response'],
                    'context_docs': data['context_docs'],
                    'timestamp': data.get('timestamp', 0)
                })
            
            json_data = {
                'document_id': document_id,
                'entries': entries,
                'total_entries': len(entries)
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(json_data, ensure_ascii=False, indent=2))
            
            logger.info(f"Metadados salvos para documento {document_id}: {len(entries)} entradas")
        except Exception as e:
            logger.error(f"Erro ao salvar metadados para {document_id}: {e}")

    def _create_faiss_index(self, query: str, document_id: str, query_hash: str) -> Optional[FAISS]:
        """Cria um índice FAISS para uma query específica"""
        try:
            # Computar embedding da query
            query_embedding = self.embedder.embed_query(query)
            
            # Criar índice FAISS com um único embedding
            texts = [query]
            embeddings = [query_embedding]
            
            faiss_index = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings)),
                embedding=self.embedder
            )
            
            # Salvar índice no disco
            faiss_path = self._get_faiss_path(document_id, query_hash)
            os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
            faiss_index.save_local(faiss_path)
            
            logger.info(f"Índice FAISS criado para query {query_hash} do documento {document_id}")
            return faiss_index
            
        except Exception as e:
            logger.error(f"Erro ao criar índice FAISS: {e}")
            return None

    def _load_faiss_index(self, document_id: str, query_hash: str) -> Optional[FAISS]:
        """Carrega um índice FAISS do disco"""
        faiss_path = self._get_faiss_path(document_id, query_hash)
        
        if not os.path.exists(faiss_path):
            return None
        
        try:
            faiss_index = FAISS.load_local(
                faiss_path, 
                self.embedder, 
                allow_dangerous_deserialization=True
            )
            return faiss_index
        except Exception as e:
            logger.error(f"Erro ao carregar índice FAISS {faiss_path}: {e}")
            return None

    def _compute_similarity_with_faiss(self, query: str, document_id: str) -> Optional[Tuple[str, str, float]]:
        """Usa FAISS para encontrar queries similares"""
        metadata = self._load_metadata(document_id)
        if not metadata:
            return None
        
        best_match = None
        best_similarity = 0
        best_query_hash = None
        
        # Limitar busca aos itens mais recentes para performance
        recent_items = list(metadata.items())[-min(20, len(metadata)):]
        
        for query_hash, data in recent_items:
            faiss_index = self._load_faiss_index(document_id, query_hash)
            if faiss_index is None:
                continue
            
            try:
                # Buscar similaridade usando FAISS
                docs_and_scores = faiss_index.similarity_search_with_score(query, k=1)
                if docs_and_scores:
                    _, score = docs_and_scores[0]
                    # FAISS retorna distância, converter para similaridade
                    similarity = 1.0 / (1.0 + score)
                    
                    if similarity > self.similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = (data['response'], data['context_docs'])
                        best_query_hash = query_hash
                        
            except Exception as e:
                logger.error(f"Erro na busca FAISS para {query_hash}: {e}")
                continue
        
        if best_match and best_query_hash:
            # Mover para o final (mais recente)
            metadata.move_to_end(best_query_hash)
            self._save_metadata(document_id)
            return (best_match[0], best_match[1], best_similarity)
        
        return None

    def get(self, query: str, document_id: str) -> Optional[Tuple[str, List[dict]]]:
        """Busca no cache com persistência em disco"""
        metadata = self._load_metadata(document_id)
        if not metadata:
            return None
        
        query_hash = self._get_query_hash(query)
        
        # Verificar hit exato primeiro
        if query_hash in metadata:
            data = metadata[query_hash]
            metadata.move_to_end(query_hash)
            self._save_metadata(document_id)
            logger.info(f"Cache hit exato para documento {document_id}")
            return (data['response'], data['context_docs'])
        
        # Busca semântica usando FAISS
        similarity_result = self._compute_similarity_with_faiss(query, document_id)
        if similarity_result:
            response, context_docs, similarity = similarity_result
            logger.info(f"Cache hit semântico para documento {document_id} com similaridade {similarity:.4f}")
            return (response, context_docs)
        
        return None

    def put(self, query: str, document_id: str, response: str, context_docs: List[dict]):
        """Adiciona item ao cache com persistência em disco"""
        if not query.strip() or not response.strip():
            return
        
        query_hash = self._get_query_hash(query)
        metadata = self._load_metadata(document_id)
        
        # Criar índice FAISS para a query
        faiss_index = self._create_faiss_index(query, document_id, query_hash)
        if faiss_index is None:
            logger.warning(f"Não foi possível criar índice FAISS para query: {query[:50]}...")
            return
        
        # Adicionar aos metadados
        import time
        metadata[query_hash] = {
            'query': query,
            'response': response,
            'context_docs': context_docs,
            'timestamp': time.time()
        }
        
        # Limitar tamanho do cache
        while len(metadata) > self.max_cache_size:
            # Remover o mais antigo
            old_query_hash = next(iter(metadata))
            old_faiss_path = self._get_faiss_path(document_id, old_query_hash)
            
            # Remover índice FAISS do disco
            try:
                if os.path.exists(old_faiss_path):
                    import shutil
                    shutil.rmtree(old_faiss_path)
                    logger.debug(f"Índice FAISS removido: {old_faiss_path}")
            except Exception as e:
                logger.error(f"Erro ao remover índice FAISS antigo: {e}")
            
            # Remover dos metadados
            metadata.popitem(last=False)
        
        # Salvar metadados
        self._save_metadata(document_id)
        logger.info(f"Item adicionado ao cache para documento {document_id}")

    def clear_document_cache(self, document_id: str):
        """Limpa cache de um documento específico"""
        doc_cache_path = self._get_document_cache_path(document_id)
        
        try:
            if os.path.exists(doc_cache_path):
                import shutil
                shutil.rmtree(doc_cache_path)
                logger.info(f"Cache em disco removido para documento {document_id}")
            
            # Limpar cache em memória
            if document_id in self._metadata_cache:
                del self._metadata_cache[document_id]
                
        except Exception as e:
            logger.error(f"Erro ao limpar cache para documento {document_id}: {e}")

    def get_cache_stats(self) -> Dict[str, int]:
        """Retorna estatísticas do cache"""
        total_entries = 0
        total_documents = 0
        
        try:
            if os.path.exists(self.questions_path):
                for doc_dir in os.listdir(self.questions_path):
                    doc_path = os.path.join(self.questions_path, doc_dir)
                    if os.path.isdir(doc_path):
                        total_documents += 1
                        metadata_path = os.path.join(doc_path, "metadata.json")
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, 'r', encoding='utf-8') as f:
                                    text = f.read()
                                    data = json.loads(text)
                                    total_entries += len(data.get('entries', []))
                            except Exception:
                                pass
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas: {e}")
        
        return {
            "total_documents": total_documents,
            "total_entries": total_entries,
            "cache_path": self.questions_path,
            "memory_cached_docs": len(self._metadata_cache)
        }

    def cleanup_old_entries(self, days_old: int = 30):
        """Remove entradas antigas do cache"""
        import time
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        for document_id in list(self._metadata_cache.keys()):
            metadata = self._load_metadata(document_id)
            removed_count = 0
            
            for query_hash in list(metadata.keys()):
                entry_time = metadata[query_hash].get('timestamp', 0)
                if entry_time < cutoff_time:
                    # Remover índice FAISS
                    faiss_path = self._get_faiss_path(document_id, query_hash)
                    try:
                        if os.path.exists(faiss_path):
                            import shutil
                            shutil.rmtree(faiss_path)
                    except Exception as e:
                        logger.error(f"Erro ao remover FAISS antigo: {e}")
                    
                    # Remover dos metadados
                    del metadata[query_hash]
                    removed_count += 1
            
            if removed_count > 0:
                self._save_metadata(document_id)
                logger.info(f"Removidas {removed_count} entradas antigas do documento {document_id}")
