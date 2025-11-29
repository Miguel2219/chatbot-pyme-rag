import sys 
import os
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from backend.rag_engine import RAGEngine

def show_banner():
    """Muestra el banner de bienvenida del chatbot."""
    print("\n")
    print("=" * 70)
    print("=                                                                    =")
    print("=        CHATBOT RESTAURANTE EL BUEN SABOR - MODO PRUEBA         =")
    print("=                                                                    =")
    print("=" * 70)
    print("\n Restaurante ubicado en Chapinero, Bogotá")
    print(" WhatsApp: 300-123-4567")
    print("\n Puedes preguntar sobre:")
    print("   - Menú y precios")
    print("   - Horarios de atención")
    print("   - Servicio de domicilios")
    print("   - Reservas y políticas")
    print("   - Preguntas frecuentes")
    print("\n  Comandos especiales:")
    print("   'salir' - Terminar")
    print("   'stats' - Ver estadísticas acumuladas")
    print("   'ayuda' - Ver esta ayuda")
    print("\n" + "=" * 70 + "\n")
    
def show_help():
    """Muestra la ayuda con comandos disponibles."""
    
    print("\n" + "=" * 70)
    print(" AYUDA - COMANDOS DISPONIBLES")
    print("=" * 70)
    print("\n Para chatear:")
    print("   Simplemente escribe tu pregunta y presiona Enter")
    print("\n Comandos especiales:")
    print("   salir - Terminar el programa")
    print("   stats - Ver estadísticas de uso (tokens, costo)")
    print("   ayuda - Mostrar esta ayuda")
    print("\n Ejemplos de preguntas:")
    print("   - ¿Cuánto cuesta el ajiaco?")
    print("   - ¿Cuál es el horario de atención?")
    print("   - ¿Hacen domicilios?")
    print("   - ¿Tienen opciones vegetarianas?")
    print("   - ¿Aceptan tarjeta de crédito?")
    print("\n" + "=" * 70 + "\n")
    
def show_stats(stats):
    print("\n" + "=" * 70)
    print(" ESTADÍSTICAS DE LA SESIÓN")
    print("=" * 70)
    print(f"\n Consultas realizadas: {stats['total_queries']}")
    print(f" Consultas exitosas: {stats['successful_queries']}")
    print(f" Consultas con error: {stats['failed_queries']}")
    
    if stats['total_queries'] > 0:
        print(f"\n  Tiempo promedio por consulta: {stats['avg_time']:.2f}s")
        print(f" Tokens totales usados: {stats['total_tokens']:,}")
        print(f" Costo total: ${stats['total_cost']:.6f} USD")
        print(f" Costo promedio por consulta: ${stats['avg_cost']:.6f} USD")
    
    print("\n" + "=" * 70 + "\n")
    
def format_response(result, show_sources=True, show_stats_inline=True):
    print("\n El Buen Sabor:")
    print("-" * 70)
    print(result['answer'])
    print("-" * 70)
    
    if show_sources and result['sources']:
        print('\nFuentes consultadas: ')
        for i, source in enumerate(result['sources'], 1):
            print(f"   {i}. {source['source']} (chunk {source['chunk_index']}) - similitud: {1 - source['distance']:.2%}")
            
    if show_stats_inline:
        print(f"\n📊 Tokens: {result['tokens_used']} | Costo: ${result['cost']:.6f} | Tiempo: {result['time_total']:.2f}s")

def interactive_chat():
    show_banner()
    try:
        engine = RAGEngine()
    except Exception as es:
        return
    stats = {
        'total_queries': 0,
        'successful_queries': 0,
        'failes_queries': 0,
        'total_tokens': 0,
        'total_cost': 0.0,
        'total_time': 0.0,
        'avg_time': 0.0,
        'avg_cost': 0.0
    }
    while True:
        try:
            user_input = input("\n Tu: ").strip()
            if not user_input:
                continue
            command = user_input.lower()
            if command == 'salir':
                if stats['total_queries'] > 0:
                    show_stats(stats)
                break
            elif command == 'ayuda':
                show_help()
                continue
            elif command == 'stats':
                show_stats(stats)
                continue
            stats['total_queries'] += 1
            result = engine.query(user_input, verbose=False)
            if 'error' in result:
                stats['failes_queries'] += 1
                print(f"\n Error: {result['error']}")
            else:
                stats['successful_queries'] += 1
                format_response(result, show_sources=True, show_stats_inline=True)
                
        except Exception:
            stats['failes_queries'] += 1
            continue
if __name__ == "__main__":
    interactive_chat()