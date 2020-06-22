<template>
    <div>
      <div>
        <div class="container">
          <div class="row">
            <div class="col-md-12">
              <div class="ptb-50">
                <h2 class="text-center">자소서를 통한 MBTI 유형 분석</h2>
                <!-- <div id="movetopbt" style="color:black">자기소개서 쓰기</div> -->
              </div>
            </div>
          </div>
        <div class="row">
            <div class="col-sm-6 col-md-4">
                  <div class="single_what_we_do">
                    <div class="top_line"></div>
                    <div class="what_we_do_figure">
                      <img src="../1.png" alt="" class="img-responsive">
                    </div>
                    <h4 class="what_we_do_title">자소서 방향 잡기</h4>
                    <div class="what_we_do_content">자신의 성향을 분석함으로써 자소서의 방향을 잡을 수 있습니다.</div>
                  </div>
                </div>
                <div class="col-sm-6 col-md-4">
                  <div class="single_what_we_do">
                    <div class="top_line"></div>
                    <div class="what_we_do_figure">
                      <img src="../2.png" alt="" class="img-responsive">
                    </div>
                    <h4 class="what_we_do_title">맞는 인재상인가?</h4>
                    <div class="what_we_do_content">기업에서 원하는 인재상에 부합하는지를 예측할 수 있습니다.</div>
                  </div>
                </div>
                <div class="col-sm-6 col-md-4">
                  <div class="single_what_we_do">
                    <div class="top_line"></div>
                    <div class="what_we_do_figure">
                      <img src="../3.png" alt="" class="img-responsive">
                    </div>
                    <h4 class="what_we_do_title">나에게 맞는 직무</h4>
                    <div class="what_we_do_content">내가 원하는 직무의 인재상과 적합한지를 확인하고 나를 되돌아볼수 있습니다.</div>
                  </div>
                </div>
        </div>
     </div>
    </div>
    <!-- 자소서 쓰는 textarea -->
    <div class="container" style="margin-top:40px;">
          <div class="row text-center">
            <div class="col-md-8 center-block" style="float:none; margin:0 auto">
              <form class="form-horizontal">
                <div class="form-group" style="text-align: center;">
                   <h4 class="what_we_do_title">나의 자소서</h4>
                  <div class="wrap">
    <textarea id="content" style="height:300px" v-model="before" placeholder="20자 이상의 자소서를 입력하세요."></textarea>
    <span id="counter">###</span>
    <span id="counter_left">###</span>
</div>
                  </div>
                 <!-- <button type="submit" class="btn btn-default" style="background-color: #DDEEEE">결과보기</button> -->
                <div style="height:80px;">
                 <!-- <div v-if="before!==''"> -->
                <div v-if="before.length>=19">
                  <b-button class="mt-3" v-on:click="hover" @click="$bvModal.show('bv-modal-example')">결과보기</b-button>
                 </div>
                </div>
                  <b-modal id="bv-modal-example" hide-footer>
                    <template v-slot:modal-title>
                      당신의 유형은?
                    </template>
                    <div class="d-block">
                      <h1 style="text-align: center">{{after.MBTI}}</h1>
                      <div v-if="after.MBTI === 'ISTJ'">
                        <h4 style="text-align: center">소금형</h4>
                        <img src="../img/ISTJ.jpg"  style="width:100%">
                        <ul>
                          <li>한번 시작한 일은 끝까지 해내는 성격.</li>
                          <li>신뢰성이 있고, 현실적, 실용적, 철저하고, 건실하고, 체계적이다.</li>
                          <li>열심히 일하고 세부적인 사항과 절차에 세심하다.</li>
                        </ul>
                        <!-- ISTJ 선호경향이 있는 사람들은 매우 신뢰성이 있고, 사실에 대한 완전하고, 현실적이고, 실용적인 면을 가지고 있습니다. 어떠한 분량의 사실이라도 몰두하고, 기억하며, 이용하고, 정확도에 대해 매우 세심합니다. 위기상황에 대처할 때에도 차분하며 안정되어 있습니다. 그들이 어떤 사람인가를 아는 데는 상당한 시간이 걸립니다. 그들은 외면적으로 볼 때 차분하게 보이지만, 내면적으로는 상당히 바쁩니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'ISFJ'">
                        <h4  style="text-align: center">전통주의형</h4>
                        <img src="../img/ISFJ.jpg"  style="width:100%">
                        <ul>
                          <li>성실하고 온화하며 협조를 잘하는 사람.</li>
                          <li>책임감이 강하고 온정적이며 헌신적이다.</li>
                          <li>치밀함과 반복을 요하는 일을 끝까지 해 나가는 인내력이 높다.</li>
                        </ul>
                        <!-- ISFJ유형은 책임감이 강하고 온정적이며 헌신적입니다. 세부적이고 치밀성과 반복을 요하는 일을 끝까지 해 나가는 인내력이 높습니다. 이들이 가진 침착성과 인내력은 가정이나 집단에 안정성을 가져다줍니다. 다른 사람의 사정을 고려하며 자신과 타인의 감정의 흐름에 민감합니다. 일처리에 있어 현실감각을 가지고 실제적이고 조직적으로 이행합니다. 경험을 통해서 자신이 틀렸다고 인정하기까지 어떠한 난관이 있어도 꾸준히 밀고 나가는 유형입니다. 온순하고 차분합니다 -->
                      </div>
                      <div v-else-if="after.MBTI === 'INFJ'">
                        <h4 style="text-align: center">예언자형</h4>
                        <img src="../img/INFJ.jpg"  style="width:100%">
                        <ul>
                          <li>창의력과 사람에 대한 통찰력이 뛰어난 유형.</li>
                          <li>확고한 신념과 뚜렷한 원리원칙을 생활 속에 가지고 있다.</li>
                          <li>공동의 이익을 가져오는 일에 심혈을 기울인다.</li>
                        </ul>
                        <!-- INFJ유형의 사람들은 창의력과 통찰력이 뛰어납니다. 강한 직관력으로 의미와 진실된 관계를 추구합니다. 뛰어난 영감으로 말없이 타인에게 영향력을 끼칩니다. 독창성과 사적인 독립심이 강합니다. 또한 확고한 신념과 뚜렷한 원리원칙을 생활 속에 가지고 있으면서 공동의 이익을 가져오는 일에 심혈을 기울입니다. 인화와 동료애를 중시하는 경향으로 존경을 받고 사람들이 따릅니다. 열정과 신념으로 자신들이 믿는 영감을 구현 시켜 나갈 때 위대한 정신적 지도자들이 이 유형에서 많이 나옵니다.  남에게 강요하기보다 행동과 권유로 사람들의 마음을 움직여 따르게 만드는 지도력이 있습니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'INTJ'">
                        <h4 style="text-align: center">과학자형</h4>
                        <img src="../img/INTJ.jpg"  style="width:100%">
                        <ul>
                          <li>전체를 조합하여 비전을 제시하는 사람.</li>
                          <li>16가지 유형 중에서 가장 독립적이고 단호하다.</li>
                          <li>직관력과 통찰력이 있는 유형.</li>
                        </ul>
                        <!-- INTJ유형은 행동과 사고에 있어 독창적입니다. 내적인 신념과 비전은 산이라도 움직일 만큼 강합니다. 16가지 유형 중에서 가장 독립적이고 단호합니다. 때때로 어떤 문제에 대하여 아주 고집에 셉니다. 자신이 가진 영감과 목적을 실현시키려는 의지와 결단력과 인내심을 가지고 있습니다. 자신과 타인의 능력을 중요시하며, 목적달성을 위하여 온 시간과 노력을 다 바쳐 일합니다. 직관력과 통찰력이 활용되는 분야에서 능력을 발휘하는 편입니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'ISTP'">
                        <h4 style="text-align: center">백과사전형</h4>
                        <img src="../img/ISTP.jpg"  style="width:100%">
                        <ul>
                          <li>논리적이고 뛰어난 상황적응력을 가지고 있다.</li>
                          <li>조용하고 말이 없다.</li>
                          <li>논리적이고 분석적이고 객관적으로 인생을 관찰하는 유형.</li>
                        </ul>
                        <!-- 조용하고 말이 없으며 논리적이고 분석적이고 객관적으로 인생을 관찰하는 유형입니다. 사실적인 정보를 조직하기 좋아하는 반면 일과 관계 되지 않는 이상 자신을 개방하지 않는 편입니다.  그래서 가까운 친구들 외에는 대체로 사람들과 사귀지 않는 편입니다. 일상생활에 있어 매우 적응력이 강합니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'ISFP'">
                        <h4 style="text-align: center">성인군자형</h4>
                        <img src="../img/ISFP.jpg"  style="width:100%">
                        <ul>
                          <li>따뜻한 감성을 가지고 있는 겸손한 사람.</li>
                          <li>적응력과 관용성이 많으며, 삶의 현재를 즐기는 유형.</li>
                          <li>마음이 따뜻하고 동정적이다.</li>
                        </ul>
                        <!-- ISFP형을 선호하는 사람들은 말보다는 행동으로 따뜻함을 나타내며, 마음이 따뜻하고 동정적입니다. 마치 양털안감을 넣은 오버코트처럼 속마음이 따뜻한 사람들입니다. 그러나 상대방을 잘 알게 될 때까지 이 따뜻함을 잘 드러내지 않는 경향이 있습니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'INFP'">
                        <h4 style="text-align: center">이상주의형</h4>
                        <img src="../img/INFP.jpg"  style="width:100%">
                        <ul>
                          <li>이상적인 세상을 꿈꾸고 만들어가는 사람들.</li>
                          <li>조용하며, 자신이 관계하는 사람이나 일에 대하여 성실하다.</li>
                          <li>완벽주의적 경향이 있다.</li>
                        </ul>
                        <!-- INFP유형은 마음이 따뜻하나, 상대방을 잘 알기 전에는 표현을 잘하지 않습니다. 조용하며, 자신이 관계하는 사람이나 일에 대하여 강하고 성실합니다. 또한 자신이 지향하는 이상에 대하여는 정열적인 신념을 지니고 있습니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'INTP'">
                        <h4 style="text-align: center">아이디어형</h4>
                        <img src="../img/INTP.jpg"  style="width:100%">
                        <ul>
                          <li>비평적인 관점을 가진 뛰어난 전략가.</li>
                          <li>조용하고 과묵하나 관심이 있는 분야에 대해서는 말을 잘한다.</li>
                          <li>인간관계나 모임보다는 실제적인 토론을 즐긴다.</li>
                        </ul>
                        <!-- INTP 선호경향이 있는 사람들은 조용하고 과묵하나 관심이 있는 분야에 대해서는 말을 잘합니다. 사람들을 중심으로 한 가치보다는 아이디어에 관심이 많으며 매우 분석적이고 논리적이며 객관적 비평을 잘합니다. 일의 원리와 아이디어에 관심이 많으며, 실체보다는 실체가 안고 있는 가능성에 관심이 많습니다. 이해가 빠르고 높은 직관력으로 통찰하는 재능과 지적 관심이 많습니다. 개인적인 인간관계나 파티 혹은 잡담 등에는 흥미가 별로 없습니다. 사람을 사귀는 데 있어서 그들은 보통 아이디어를 토론하고 나누는 소수의 가까운 사람들을 주위에 두고 있습니다. 때로 어떤 아이디어에 몰입하여 주위에서 돌아가고 있는 일을 모를 때가 많습니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'ESTP'">
                        <h4 style="text-align: center">활동가형</h4>
                        <img src="../img/ESTP.jpg"  style="width:100%">
                        <ul>
                          <li>친구, 운동, 음식 등 다양함을 선호.</li>
                          <li>관대하고 느긋하다.</li>
                          <li>선입관이 적고 개방적이다.</li>
                          <li>문제해결사 역할을 잘한다.</li>
                        </ul>
                        <!-- ESTP 선호경향을 지닌 사람들은 관대하고 느긋하며 어떤 사람이나 사건에 대해서 별로 선입관을 갖지 않으며 개방적입니다. 자신에게나 타인에게 있어 관용적이며, 일을 있는 그대로 보고 받아들입니다. 그래서 갈등이나 긴장이 일어나는 상황을 잘 무마하는 능력이 있습니다. ‘꼭 이렇게 되어야 하고, 저렇게 되어야 된다.’라는 규범을 적용하기보다 누구나 만족할 수 있는 해결책을 모색합니다. 또한 타협하고 적응하는 능력이 있습니다. 주의집중을 현재 상황에 맞추고, 현실을 있는 그대로 보는 이 유형의 자연스러운 경향으로 인해 현실적으로 야기되는 문제 해결에 뛰어난 능력을 발휘하기도 합니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'ESFP'">
                        <h4 style="text-align: center">사교형</h4>
                        <img src="../img/ESFP.jpg"  style="width:100%">
                        <ul>
                          <li>분위기를 고조시키는 우호적이고 발랄한 성격.</li>
                          <li>친절하고 수용적이며 현실적이다.</li>
                          <li>실제적으로 물질적 소유 및 운동을 비롯하여 실생활을 즐기는 유형.</li>
                        </ul>
                        <!-- ESFP유형은 친절하고 수용적이며 현실적이고 실제적입니다. 어떤 상황에도 잘 적응하며 타협적입니다. 선입견이 별로 없으며 개방적이고 관용적이며 대체로 사람들을 잘 받아들입니다. 주위에서 진행되고 있는 일들을 알고자 하고 또한 열심히 참견하고 끼어들곤 합니다. 다른 사람의 일이나 활동에 관심이 많으며 새로운 사건 혹은 물건에도 관심과 호기심이 많기 때문입니다. 이론이나 책을 통해 배우기보다 경험 등의 실생활을 통해 배우는 것을 선호합니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'ENFP'">
                        <h4 style="text-align: center">스파크형</h4>
                        <img src="../img/ENFP.jpg"  style="width:100%">
                        <ul>
                          <li>열정적으로 새 관계를 만들어가는 사람.</li>
                          <li>풍부한 상상력과 영감을 가지고 새로운 프로젝트를 잘 시작한다.</li>
                          <li>즉흥적으로 일을 시작하고 재빠르게 해결하는 솔선수범력.</li>
                        </ul>
                        <!-- ENFP 유형은 열성적이고 창의적입니다. 풍부한 상상력과 영감을 가지고 새로운 프로젝트를 잘 시작합니다. 풍부한 충동적 에너지를 가지고 즉흥적으로 일을 재빠르게 해결하는 솔선수범력과 상상력이 있습니다. 관심이 있는 일이면 무엇이든 척척 해내는 열성파입니다. 뛰어난 통찰력으로 그 사람 안에 있는 성장 발전할 가능성을 들여다봅니다. 또한 자신의 열성으로 다른 사람들도 어떤 프로젝트에 흥미를 가지게 하고 다른 사람을 잘 도와줍니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'ENTP'">
                        <h4 style="text-align: center">발명가형</h4>
                        <img src="../img/ENTP.jpg"  style="width:100%">
                        <ul>
                          <li>풍부한 상상력으로 새로운 것에 도전한다.</li>
                          <li>독창적인 혁신가이며, 창의력이 풍부하다.</li>
                          <li>항상 새로운 가능성을 찾고 새로운 시도를 즐기는 유형.</li>
                        </ul>
                        <!-- ENTP유형은 독창적인 혁신가이며, 창의력이 풍부하고 항상 새로운 가능성을 찾고 새로운 시도를 즐기는 유형입니다. 넓은 안목을 가지고 있으며 다방면에 재능이 있습니다. 풍부한 상상력과 어떤 프로젝트를 시도하는 솔선력이 강합니다. 민첩하고 여러 가지 일에 재능을 발휘하며 자신감이 넘쳐 납니다. 사람들의 동향에 대해 기민하고 박식한 편입니다. 다른 사람을 판단하기보다 이해하려고 노력합니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'ESTJ'">
                        <h4 style="text-align: center">사업가형</h4>
                        <img src="../img/ESTJ.jpg"  style="width:100%">
                        <ul>
                          <li>사무적, 실용적, 현실적인 스타일.</li>
                          <li>현실적이고 사실적이며 체계적, 논리적이다.</li>
                        </ul>
                        <!-- ESTJ 선호경향이 있는 사람들은 일을 조직하고 프로젝트를 계획하고 출범시키는 능력이 뛰어납니다. 현실적이고 사실적이며 체계적, 논리적으로 사업이나 조직체를 이끌어가는 타고난 재능을 가졌기 때문입니다. 혼돈스러운 상태나 불분명한 상태 또는 실용성이 없는 분야에는 큰 흥미가 없습니다. 그러나 필요시에는 언제나 응용하는 힘이 있습니다. 분명한 규칙을 중요시하고 그에 따라 행동하고 일을 추진하고 완성해 나갑니다. 어떤 계획이나 결정을 내릴 때 확고한 사실에 바탕을 두고 이행합니다. 대체로 결과를 현재 볼 수 있는 일을 즐기는 편입니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'ESFJ'">
                        <h4 style="text-align: center">친선도모형</h4>
                        <img src="../img/ESFJ.jpg" style="width:100%">
                        <ul>
                          <li>친절, 현실감을 바탕으로 타인에게 봉사하는 유형.</li>
                          <li>동정심과 동료애가 많다.</li>
                          <li>친절하고 재치가 있다.</li>
                          <li>다른 사람들에게 관심이 많고 잘 돕는다.</li>
                        </ul>
                        <!-- ESFJ유형은 유난히 동정심과 동료애가 많습니다. 친절하고 재치가 있으며, 다른 사람들에게 관심을 쏟고 인화를 도모하는 일을 중요하게 여깁니다. 양심적이고 정리정돈을 잘합니다. 참을성이 많고 다른 사람들을 잘 돕습니다. 또한 다른 사람들의 지지를 받으면 일에 열중합니다. 그러나 다른 사람들의 무관심한 태도에는 민감합니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'ENFJ'">
                        <h4 style="text-align: center">언변능숙형</h4>
                        <img src="../img/ENFJ.jpg"  style="width:100%">
                        <ul>
                          <li>타인의 성장을 도모하고 협동하는 스타일.</li>
                          <li>동정심과 동료애가 많으며, 친절하고 재치있다.</li>
                          <li>민첩하고 참을성이 많고 성실하다.</li>
                        </ul>
                        <!-- ENFJ 유형은 동정심과 동료애가 많으며, 친절하고 재치 있습니다. 민첩하고 참을성이 많고 성실합니다. 사람들과의 인화를 중요시 여기며, 다른 사람들의 의견을 존중하고 그 가치를 봅니다. 공동의 선(善)을 위해서는 대체로 상대방 의견에 동의하는 경향이 있습니다. 새로운 아이디어에 대한 호기심이 많습니다. 쓰기 보다는 말로써 생각을 잘 표현합니다. 편안하고 능란하게 계획을 제시하거나 조직을 이끌어 가는 능력이 있습니다. -->
                      </div>
                      <div v-else-if="after.MBTI === 'ENTJ'">
                        <h4 style="text-align: center">지도자형</h4>
                        <img src="../img/ENTJ.jpg"  style="width:100%">
                        <ul>
                          <li>비전을 갖고 타인을 활력적으로 인도하는 유형.</li>
                          <li>활동적이며 행정적인 일과 장기계획을 선호한다.</li>
                          <li>논리적이고 분석적이다.</li>
                          <li>목적달성을 추진시키는 지도자형.</li>
                        </ul>
                        <!-- ENTJ 유형은 활동적이며 행정적인 일과 장기계획을 선호하며 논리적이고 분석적입니다. 사전 준비를 철저히 하며 계획하고 조직하고 체계적으로 목적달성을 추진시키는 지도자들이 많습니다. 비능률적이거나 확실치 않는 상황에 대해서는 별로 인내심이 없습니다. 그러나 상황이 필요로 할 때는 강하게 대처합니다. 솔직하고 결정력과 통솔력이 있으며, 거시적 안목으로 일을 밀고 나갑니다. 관념 자체에 집중하는 경향이 있으며, 관념 이면의 사람에게는 관심이 별로 없습니다. -->
                      </div>
                      <!-- <div id='chart' style = "width:100%">
                        <div style = "width:100%;">
                        <pie-chart :data="chartData" :options="chartOptions"></pie-chart>
                        </div>
                      </div> -->
                      <!-- <div>{{resultMessage}}</div>
                      <div class="row" style="margin-left:5%">
                        <div class="col-6">E : {{after.E}}</div>
                        <div class="col-6">F : {{after.F}}</div>
                        <div class="col-6">I : {{after.I}}</div>
                        <div class="col-6">J : {{after.J}}</div>
                        <div class="col-6">N : {{after.N}}</div>
                        <div class="col-6">P : {{after.P}}</div>
                        <div class="col-6">S : {{after.S}}</div>
                        <div class="col-6">T : {{after.T}}</div>
                      </div> -->
                      <!-- <br>
                      <div>{{after}}</div> -->
                    </div>
                    <b-button class="mt-3" block v-on:click="detail">상세보기</b-button>
                    <b-button class="mt-3" block @click="$bvModal.hide('bv-modal-example')">Close Me</b-button>
                  </b-modal>
              </form>
              <img src = "../img/human.png" style="height:250px;">
            </div>
          </div>
        </div>
  </div>
</template>

<script>

import common from '../http_common'
import PieChart from '../PieChart.js'
import $ from 'jquery'
// import axios from 'axios'
export default {
  name: 'Login',
  components: {
    PieChart
  },
  data () {
    return {
      episodes: '',
      result: '',
      x: '',
      y: '',
      before: '',
      after: '',
      EE: '',
      credentials: {
        username: '',
        password: ''
      },
      error_msg: '',
      chartOptions: {
        hoverBorderWidth: 20
      },
      chartData: {
        hoverBackgroundColor: 'red',
        hoverBorderWidth: 10,
        labels: ['E', 'F', 'I', 'J', 'N', 'P', 'S', 'T'],
        datasets: [
          {
            label: 'Data One',
            backgroundColor: ['#41B883', '#E46651', '#00D8FF', '#ff9be6', '#ffff00', '#ff9500', '#ae14ac', '#000000'],
            data: [0.5, 0.5]
            // data: [this.EE, 'this.EE', '{{this.EE}}'
            // data: [this.EE]
          }
        ]
      }
    }
  },
  mounted () {
    $(function () {
      $('#content').keyup(function (e) {
        this.content = $(this).val()
        // $(this).height(((content.split('\n').length + 1) * 1.5) + 'em')
        this.content_length = this.content.length
        if (this.content.length < 20) {
          $('#counter_left').html('20자이상 입력하세요.')
          $('#counter').html(this.content.length + '/1500')
        } else {
          $('#counter_left').html('')
          $('#counter').html(this.content.length + '/1500')
        }
      })
      $('#content').keyup()
    })
    $(document).ready(function () {
      $('#movetopbt').bind('click', function () {
        $('html, body').animate({scrollTop: $('#content').offset().top}, 680)
      })
    })
  },
  computed: {
    resultMessage: function () {
      return this.after
    }
  },
  watch: {
    resultMessage: {
      deep: true,
      handler: function (newVal) {
        this.chartData.datasets[0].data = [this.after.E, this.after.I]
        localStorage.setItem('I', this.after.I)
        localStorage.setItem('E', this.after.E)
        localStorage.setItem('S', this.after.S)
        localStorage.setItem('N', this.after.N)
        localStorage.setItem('T', this.after.T)
        localStorage.setItem('F', this.after.F)
        localStorage.setItem('J', this.after.J)
        localStorage.setItem('P', this.after.P)
        localStorage.setItem('EI', this.after.EI)
        localStorage.setItem('JP', this.after.JP)
        localStorage.setItem('SN', this.after.SN)
        localStorage.setItem('TF', this.after.TF)
      }
    }
  },
  methods: {
    detail: function () {
      this.$router.push('/detail')
    },
    hover: function () {
      this.trans()
    },
    post: function () {
      common.post('/post', this.x)
        .then((res) => {
          this.y = res.data
        })
    },
    trans: function () {
      console.log(1)
      common.post('/trans', this.before)
        .then((res) => {
          this.after = res.data
          this.EE = res.data.E
        })
      console.log(2)
    },
    login: function () {
      common.get('/click')
        .then((res) => {
          this.result = res.data
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        })
    }
  },
  created () {
    // this.getEps()
  }
}
</script>

<style scoped>
.wrap {
    width: 100%;
    height: 300px;
    position: relative;
    display: inline-block;
}
.wrap textarea {
    width: 100%;
    height: 300px;
    resize: none;
    /* min-height: 4.5em;
    line-height:1.6em;
    max-height: 9em; */
}
.wrap span {
    position: absolute;
    bottom: 3px;
    right: 7px;
}
#counter_left{
    position: absolute;
    bottom: 3px;
    left: 7px;
}
.ptb-50{
    padding: 50px 50px;
}

.section_bg_gray {
    background-color: #DDEEEE;
}

.single_what_we_do {
    background-color: #ffffff;
    text-align: center;
    padding: 0 35px;
    border-radius: 10px;
    position: relative;
    margin-bottom: 30px;
    -webkit-transition: all 0.3s ease;
    -o-transition: all 0.3s ease;
    transition: all 0.3s ease;
    background-color: rgba( 255, 255, 255, 0.5 );
}
.single_what_we_do:hover {
    -webkit-box-shadow: 0 8px 12px 0 rgba(0, 0, 0, 0.2);
    -moz-box-shadow: 0 8px 12px 0 rgba(0, 0, 0, 0.2);
    box-shadow: 0 8px 12px 0 rgba(0, 0, 0, 0.2);
}

.single_what_we_do .top_line {
    position: absolute;
    top: 0;
    left: 44%;
    background-color: rgb(103, 105, 231);
    width: 12%;
    height: 3px;
}

.single_what_we_do .what_we_do_figure {
    padding: 40px 0;
}
.single_what_we_do .what_we_do_figure img {
    margin: 0 auto;
    width: 28%;
    height: 28%;
}
.single_what_we_do .what_we_do_title {
    font-weight: 600;
    font-size: 22px;
    padding-bottom: 35px;
    margin: 0;
    color: #2b353a;
}
.single_what_we_do .what_we_do_content {
    padding-bottom: 65px;
    color: #2b353a;
}
 .single_what_we_do .what_we_do_icon {
    position: absolute;
    bottom: -9px;
    left: 47%;
    color: #fff;
    background-color: #8448ff;
    width: 30px;
    height: 30px;
    text-align: center;
    border-radius: 50%;
    -webkit-transition: all 0.3s ease;
    -o-transition: all 0.3s ease;
    transition: all 0.3s ease;
}

.single_what_we_do .what_we_do_icon i {
    font-size: 24px;
    text-align: center;
    padding: 2px 3px 2px 5px;
}

#chart {
    font-family: "Avenir", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-align: center;
    color: #2c3e50;
    margin-top: 60px;
}
</style>
