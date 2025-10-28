from itertools import chain
import os
from dotenv import load_dotenv
import requests
import time
import threading
import functools
import concurrent.futures
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os

load_dotenv()


# --- Resilience helpers: simple circuit breaker + timeout decorator ---
class CircuitBreakerError(RuntimeError):
    pass


class CircuitBreaker:
    """A very small in-process circuit breaker.

    Parameters:
      failure_threshold: number of consecutive failures to open the circuit
      recovery_timeout: seconds to wait before attempting a half-open trial
      expected_exception: exception or tuple of exceptions that count as failures
    """

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self._failure_count = 0
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._opened_since = None
        self._lock = threading.Lock()

    def _open(self):
        self._state = "OPEN"
        self._opened_since = time.monotonic()

    def _close(self):
        self._state = "CLOSED"
        self._failure_count = 0
        self._opened_since = None

    def _maybe_transition(self):
        if self._state == "OPEN" and self._opened_since is not None:
            if time.monotonic() - self._opened_since >= self.recovery_timeout:
                self._state = "HALF_OPEN"

    def call(self, func, *args, **kwargs):
        with self._lock:
            self._maybe_transition()
            if self._state == "OPEN":
                raise CircuitBreakerError("Circuit is open; skipping call")

        try:
            result = func(*args, **kwargs)
        except self.expected_exception as e:
            with self._lock:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._open()
            raise
        else:
            with self._lock:
                # on success, if half-open then close; reset failure count
                if self._state in ("HALF_OPEN", "OPEN"):
                    self._close()
                else:
                    self._failure_count = 0
            return result

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper


def timed(timeout: float):
    """Decorator to run a function in a thread and raise TimeoutError on timeout."""

    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(func, *args, **kwargs)
                try:
                    return fut.result(timeout=timeout)
                except concurrent.futures.TimeoutError as e:
                    # cancel is best-effort; underlying thread may continue until completion
                    fut.cancel()
                    raise TimeoutError(f"Function call timed out after {timeout} seconds") from e

        return wrapper

    return deco


# configure a circuit breaker for local-ollama checks and LLM invokes
_OLLAMA_CB = CircuitBreaker(failure_threshold=3, recovery_timeout=20.0, expected_exception=Exception)
_LLM_CB = CircuitBreaker(failure_threshold=2, recovery_timeout=30.0, expected_exception=Exception)


@_OLLAMA_CB
@timed(2.0)
def detect_local_ollama(port: int = 11434, timeout: float = 1.5) -> str | None:
    """Try to detect a local Ollama HTTP server.

    Returns the base URL if found (e.g. 'http://localhost:11434'), otherwise None.
    The function will try a couple of likely endpoints and return as soon as one responds.
    """
    host = "http://localhost"
    base = f"{host}:{port}"
    candidates = ["/v1/models", "/ping", "/"]
    for path in candidates:
        try:
            url = base + path
            resp = requests.get(url, timeout=timeout)
            # any 2xx/3xx/4xx response means something is listening there
            if resp.status_code >= 200:
                return base
        except requests.RequestException:
            continue
    return None
 



def main():
    print("Hello from langchain-learning!")
    print(f"API Key: {os.getenv('OPENAI_API_KEY')}")

    # Determine Ollama base URL: prefer explicit env var, otherwise try localhost detection
    ollama_env = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_HOST")
    if ollama_env:
        print(f"Using Ollama base URL from environment: {ollama_env}")
        ollama_base = ollama_env
    else:
        detected = detect_local_ollama()
        if detected:
            ollama_base = detected
            # export for libraries that read env vars
            os.environ.setdefault("OLLAMA_URL", ollama_base)
            print(f"Detected local Ollama at {ollama_base}; set OLLAMA_URL environment variable.")
        else:
            ollama_base = None
            print("No local Ollama detected (tried http://localhost:11434). If you run Ollama locally, set OLLAMA_URL env var to point at it.")
    information = """
Sachin Ramesh Tendulkar (/ˌsʌtʃɪn tɛnˈduːlkər/ ⓘ; Marathi: [sətɕin t̪eɳɖulkəɾ]; born 24 April 1973) is an Indian former international cricketer who captained the Indian national team. Often dubbed the "God of Cricket" in India, he is widely regarded as one of the greatest cricketers of all time as well as one of the greatest batsmen of all time.[5] He holds several world records, including being the all-time highest run-scorer in cricket,[6] receiving the most player of the match awards in international cricket,[7] and being the only batsman to score 100 international centuries.[8] Tendulkar was a Member of Parliament, Rajya Sabha by presidential nomination from 2012 to 2018.[9][10]

Tendulkar took up cricket at the age of eleven, made his Test match debut on 15 November 1989 against Pakistan in Karachi at the age of sixteen, and went on to represent Mumbai domestically and India internationally for over 24 years.[11] In 2002, halfway through his career, Wisden ranked him the second-greatest Test batsman of all time, behind Don Bradman, and the second-greatest ODI batsman of all time, behind Viv Richards.[12] The same year, Tendulkar was a part of the team that was one of the joint-winners of the 2002 ICC Champions Trophy. Later in his career, Tendulkar was part of the Indian team that won the 2011 Cricket World Cup, his first win in six World Cup appearances for India.[13] He had previously been named "Player of the Tournament" at the 2003 World Cup.

Tendulkar has received several awards from the government of India: the Arjuna Award (1994), the Khel Ratna Award (1997), the Padma Shri (1998), and the Padma Vibhushan (2008).[14][15] After Tendulkar played his last match in November 2013, the Prime Minister's Office announced the decision to award him the Bharat Ratna, India's highest civilian award.[16][17] He was the first sportsperson to receive the award and, as of 2024, is the youngest recipient.[18][19][20]Having retired from ODI cricket in 2012,[21][22] he retired from all forms of cricket in November 2013 after playing his 200th Test match.[23] Tendulkar played 664 international cricket matches in total, scoring 34,357 runs.[24] In 2013, Tendulkar was included in an all-time Test World XI to mark the 150th anniversary of Wisden Cricketers' Almanack, and he was one of only two specialist batsmen of the post–World War II era, along with Viv Richards, to get featured in the team.[25]

Tendulkar is regarded as a symbol of national pride in India for his achievements. In 2010, Time included Tendulkar in its annual list of the most influential people in the world.[26] Tendulkar was awarded the Sir Garfield Sobers Trophy for cricketer of the year at the 2010 International Cricket Council (ICC) Awards.[27] In 2019, he was inducted into the ICC Cricket Hall of Fame.[28]

Early life and background
Tendulkar was born at the Nirmal Nursing Home in the Dadar neighbourhood of Bombay, Maharashtra on 24 April 1973[29][30] into a Maharastrian family.[31] His father, Ramesh Tendulkar, was a Marathi-language novelist and poet while his mother, Rajni, worked in the insurance industry.[32] Tendulkar's father named him after his favourite music director, Sachin Dev Burman.[33] Tendulkar has three older siblings: two half-brothers Nitin and Ajit, and a half-sister Savita. They were his father's children by his first wife, who died after the birth of her third child.[34][35] His brother Ajit played in Bombay's Kanga Cricket League.[36]

Tendulkar spent his formative years in the Sahitya Sahawas Cooperative Housing Society in Bandra (East). As a young boy, Tendulkar was considered a bully, and he often picked fights with new children in his school.[37]

As a child, Tendulkar was interested in both tennis and cricket.[38] He particularly idolised American player John McEnroe, and emulated his hero by growing his hair long at the age of 7 or 8 years. At this time, Tendulkar also regularly wore tennis wristbands and headbands and carried a tennis racquet with him as a sign of his love for tennis.[39][40][41]

To help curb his bullying tendencies, his elder brother Ajit introduced Tendulkar to cricket in 1984. Ajit introduced him to cricket coach Ramakant Achrekar at Shivaji Park in Dadar. At their first meeting, Tendulkar did not play well. Ajit told Achrekar that he was feeling self-conscious due to the coach observing him, and was not displaying his natural game. Ajit requested the coach to give him another chance at playing, but watch while hiding behind a tree. This time, Tendulkar, apparently unobserved, played much better and was accepted at Achrekar's academy.[42][better source needed]

Achrekar was impressed with Tendulkar's talent and advised him to shift his schooling to Sharadashram Vidyamandir School,[29] a school in Dadar that had produced many notable cricketers. He made his debut as a cricketer for Sharadashram in late 1984.[43] Prior to this, Tendulkar had attended the Indian Education Society's New English School in Bandra (East).[43] He was also coached under the guidance of Achrekar at Shivaji Park in the mornings and evenings.[44] Tendulkar would practice for hours; if he became exhausted, Achrekar would put a one-rupee coin on the top of the stumps, and the bowler who dismissed Tendulkar would get the coin. If Tendulkar completed the session without getting dismissed, the coach would give him the coin. Tendulkar considers the 13 coins he won among his most prized possessions.[45] While he was training at Shivaji Park, he moved in with his aunt and uncle, who lived near the park.[43]

Besides school cricket, Tendulkar also played club cricket. In 1984, at age 11, he debuted in the Kanga Cricket League while playing for the John Bright Cricket Club.[43][46] Beginning in 1988, he played for the Cricket Club of India.[46][47]

In 1987, at the age of 14, he attended the MRF Pace Foundation in Madras (now Chennai) to train as a fast bowler, but the trainer, Australian fast bowler Dennis Lillee, was unimpressed and suggested that Tendulkar focus on his batting instead.[48] On 20 January 1987, he was a substitute for Imran Khan's side in an exhibition match at Brabourne Stadium in Bombay.[49] A couple of months later, former Indian batsman Sunil Gavaskar gave Tendulkar a pair of his own lightweight pads and told him to not get disheartened for not receiving the Bombay Cricket Association's Best Junior Cricketer Award. Of this experience, Tendulkar later said, "It was the greatest source of encouragement for me".[50][51] Tendulkar served as a ball boy in the 1987 Cricket World Cup when India played against England in the semifinal in Bombay.[52][53]

In 1988, while playing for Sharadashram, Tendulkar and Vinod Kambli batted in an unbroken 664-run partnership in a Lord Harris Shield inter-school game against St. Xavier's High School. Tendulkar scored 326 (not out) in that match and scored over 1,000 runs in the tournament.[54] This was a record partnership in any form of cricket until 2006, when it was broken by two junior cricketers in Hyderabad, India.[55]

Early career
On 14 November 1987, at age 14, Tendulkar was selected to represent Bombay in the Ranji Trophy for the 1987–88 season, but he was not selected for the final eleven in any of the matches, though he was often used as a substitute fielder.[43] A year later, on 11 December 1988, aged 15 years and 232 days, Tendulkar made his debut for Bombay against Gujarat at Wankhede Stadium and scored 100 (not out) in that match, making him the youngest Indian to score a century on debut in first-class cricket.[56] He was selected to play for the team by Bombay captain Dilip Vengsarkar, who watched him play Kapil Dev in Wankhede Stadium's cricket practice nets,[29] where the Indian team had come to play against the touring New Zealand team. Tendulkar followed this by scoring a century each in his Deodhar and Duleep Trophy debuts, which are also India's domestic cricket tournaments.[57]

Tendulkar finished the 1988–89 Ranji Trophy season as Bombay's highest run-scorer. He scored 583 runs at an average of 67.77 and was the eighth-highest run-scorer overall.[58] In both 1988 and 1989, he was picked for a young Indian team to tour England under the Star Cricket Club banner.[59] In the 1990–91 Ranji Trophy final, which Bombay narrowly lost to Haryana, Tendulkar's 96 from 75 balls was key to giving Bombay a chance of victory as it attempted to chase 355 from only 70 overs on the final day.[60]

At the start of the 1989–90 season, while playing for Rest of India, Tendulkar scored an unbeaten century in an Irani Trophy match against Delhi.[61]

In the final of 1995 Ranji Trophy, Tendulkar, captaining Bombay, scored 140 and 139 versus Punjab.[62]

In the 1995–96 Irani Cup, he captained Mumbai against Rest of India.[62] His first double century (204*) was for Mumbai while playing against the visiting Australian team at the Brabourne Stadium in 1998.[29][63] He is the only player to score a century on debut in all three of his domestic first-class tournaments (the Ranji, Irani, and Duleep Trophies).[64] Another double century was an innings of 233* against Tamil Nadu in the semi-finals of the 2000 Ranji Trophy, which he regards as one of the best innings of his career.[65][66][67]

In total, Tendulkar was part of five Ranji Trophy finals, in which Mumbai won 4.[62]

County cricket
In 1992, at the age of 19, Tendulkar became the first overseas-born player to represent Yorkshire, which, prior to Tendulkar joining the team, never selected players, even UK-based, from outside Yorkshire.[29][Note 1] Selected for Yorkshire as a replacement for the injured Australian fast bowler Craig McDermott, Tendulkar played 16 first-class matches for the team and scored 1,070 runs at an average of 46.52.[68]

Career
Further information: List of international cricket centuries by Sachin Tendulkar
Early tours
Raj Singh Dungarpur is credited for the selection of Tendulkar for the Indian tour of Pakistan in late 1989,[69] after one first class season.[70] The Indian selection committee had shown interest in selecting Tendulkar for the tour of the West Indies held earlier that year, but eventually did not select him, as they did not want him to be exposed to the dominant fast bowlers of the West Indies so early in his career.[citation needed]

Tendulkar was the youngest player to debut for India in Tests at the age of 16 years and 205 days, and also the youngest player to debut for India in ODI at the age of 16 years and 238 days.[71][72] Tendulkar made his Test debut against Pakistan in Karachi in November 1989 aged 16 years and 205 days. He scored 15 runs, being bowled by Waqar Younis, who also made his debut in that match. He was noted for how he handled numerous blows to his body at the hands of the Pakistani pace attack.[73] In the fourth and final Test match in Sialkot, he was hit on the nose by a bouncer bowled by Younis, but he declined medical assistance and continued to bat even as he his nose gushed blood.[74] In a 20-over exhibition game in Peshawar, held in parallel with the bilateral series, Tendulkar made 53 runs off 18 balls, including an over in which he scored 27 runs bowled by leg-spinner Abdul Qadir.[75] This was later called "one of the best innings I have seen" by the then Indian captain Krishnamachari Srikkanth.[76] In all, Tendulkar scored 215 runs at an average of 35.83 in the Test series, and was dismissed without scoring a run in the only One Day International (ODI) he played.[77][78]

The series was followed by a tour of New Zealand in which he scored 117 runs at an average of 29.25 in Tests.[79] He was dismissed without scoring in one of the two ODI games he played, and scored 36 in the other.[80] On a 1990 tour to England, on 14 August, he became the second-youngest cricketer to score a Test century as he made 119 not out in the second Test at Old Trafford in Manchester.[74] Wisden described his innings as "a disciplined display of immense maturity" and also wrote, "He looked the embodiment of India's famous opener, Gavaskar, and indeed was wearing a pair of his pads. While he displayed a full repertoire of strokes in compiling his maiden Test hundred, most remarkable were his off-side shots from the back foot. Though only 5ft 5in tall, he was still able to control without difficulty short deliveries from the English pacemen".[81]

Tendulkar's reputation grew during the 1991–92 tour of Australia held before the 1992 Cricket World Cup. During the tour, he scored an unbeaten 148 in the third Test at Sydney, making him the youngest batsman to score a century in Australia. He then scored 114 on a fast, bouncing pitch in the final Test at Perth against a pace attack from Merv Hughes, Bruce Reid, and Craig McDermott. Hughes commented to Allan Border at the time that "This little prick's going to get more runs than you, AB."[82]

Rise through the ranks
1994–96: ODI matches
Tendulkar opened the batting for the first time in ODIs at Auckland against New Zealand in 1994, scoring an explosive 82 runs off just 49 balls.[83] This was an innings hailed by Wisden as one that “changed ODI cricket forever.”[84] He scored his first ODI century on 9 September 1994 against Australia in Sri Lanka at Colombo, in his 79th ODI.[85][86][87]
    """
    summary_template = """
    You are given biographical text about a single person.

STRICT INSTRUCTIONS:
- Use ONLY the information in {information}. Do NOT use prior knowledge.
- If a detail is not present in {information}, say "Not specified".
- Keep the person consistent with {information}. Do NOT switch to anyone else.
- Output four sections exactly in this order:
  1. Short Summary
  2. Two Interesting Facts
  3. List of Achievements
  4. Early Life and Career (brief)

Now write the four sections.
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )
    #llm = ChatOpenAI(temperature=0,model="gpt-4o-mini")
    # Try to pass the detected base url to ChatOllama if the client supports it.
    if ollama_base:
        try:
            llm = ChatOllama(temperature=0, model="gemma3:latest", base_url=ollama_base,num_ctx=16834, model_kwargs={
        "gpu_layers": 16,          # how many transformer layers on GPU
        "num_ctx": 16384,          # 8k–32k is a good default
        "num_thread": os.cpu_count(),
        "num_predict": 512,
        "keep_alive": "30m",
        "num_gpu": 1               # optional; single-GPU box
    })
        except TypeError:
            # Some versions of the client may not accept base_url param; rely on OLLAMA_URL env var instead
            llm = ChatOllama(temperature=0, model="gemma3:latest",num_ctx=16834, model_kwargs={
        "gpu_layers": 16,          # how many transformer layers on GPU
        "num_ctx": 16384,          # 8k–32k is a good default
        "num_thread": os.cpu_count(),
        "num_predict": 512,
        "keep_alive": "30m",
        "num_gpu": 1               # optional; single-GPU box
    })
    else:
        llm = ChatOllama(temperature=0, model="gemma3:latest",num_ctx=16834, model_kwargs={
        "gpu_layers": 16,          # how many transformer layers on GPU
        "num_ctx": 16384,          # 8k–32k is a good default
        "num_thread": os.cpu_count(),
        "num_predict": 512,
        "keep_alive": "30m",
        "num_gpu": 1               # optional; single-GPU box
    })

    llm_prompt = summary_prompt_template | llm

    @timed(20.0)
    @_LLM_CB
    def _invoke(prompt, input_dict):
        print("Invoking LLM...")
        return prompt.invoke(input=input_dict)

    try:
        summary = _invoke(llm_prompt, {"information": information})
    except CircuitBreakerError as e:
        summary = f"LLM circuit open or prevented call: {e}"
    except TimeoutError as e:
        summary = f"LLM invocation timed out: {e}"
    except Exception as e:
        summary = f"LLM invocation failed: {e}"
    print("Summary:")
    print(summary.content)
if __name__ == "__main__":
    main()
