(()=>{var n={188:(n,r,e)=>{"use strict";function t(n,r){return n.parent===r.parent?1:2}function i(n,r){return n+r.x}function o(n,r){return Math.max(n,r.y)}function u(){var n=t,r=1,e=1,u=!1;function a(t){var a,f=0;t.eachAfter((function(r){var e=r.children;e?(r.x=function(n){return n.reduce(i,0)/n.length}(e),r.y=function(n){return 1+n.reduce(o,0)}(e)):(r.x=a?f+=n(r,a):0,r.y=0,a=r)}));var c=function(n){for(var r;r=n.children;)n=r[0];return n}(t),h=function(n){for(var r;r=n.children;)n=r[r.length-1];return n}(t),l=c.x-n(c,h)/2,p=h.x+n(h,c)/2;return t.eachAfter(u?function(n){n.x=(n.x-t.x)*r,n.y=(t.y-n.y)*e}:function(n){n.x=(n.x-l)/(p-l)*r,n.y=(1-(t.y?n.y/t.y:1))*e})}return a.separation=function(r){return arguments.length?(n=r,a):n},a.size=function(n){return arguments.length?(u=!1,r=+n[0],e=+n[1],a):u?null:[r,e]},a.nodeSize=function(n){return arguments.length?(u=!0,r=+n[0],e=+n[1],a):u?[r,e]:null},a}function a(n){var r=0,e=n.children,t=e&&e.length;if(t)for(;--t>=0;)r+=e[t].value;else r=1;n.value=r}function f(n,r){n instanceof Map?(n=[void 0,n],void 0===r&&(r=h)):void 0===r&&(r=c);for(var e,t,i,o,u,a=new s(n),f=[a];e=f.pop();)if((i=r(e.data))&&(u=(i=Array.from(i)).length))for(e.children=i,o=u-1;o>=0;--o)f.push(t=i[o]=new s(i[o])),t.parent=e,t.depth=e.depth+1;return a.eachBefore(p)}function c(n){return n.children}function h(n){return Array.isArray(n)?n[1]:null}function l(n){void 0!==n.data.value&&(n.value=n.data.value),n.data=n.data.data}function p(n){var r=0;do{n.height=r}while((n=n.parent)&&n.height<++r)}function s(n){this.data=n,this.depth=this.height=0,this.parent=null}function d(n){return null==n?null:y(n)}function y(n){if("function"!=typeof n)throw new Error;return n}function v(){return 0}function x(n){return function(){return n}}e.r(r),e.d(r,{Node:()=>s,cluster:()=>u,hierarchy:()=>f,pack:()=>$,packEnclose:()=>B,packSiblings:()=>D,partition:()=>U,stratify:()=>W,tree:()=>un,treemap:()=>ln,treemapBinary:()=>pn,treemapDice:()=>G,treemapResquarify:()=>dn,treemapSlice:()=>an,treemapSliceDice:()=>sn,treemapSquarify:()=>hn}),s.prototype=f.prototype={constructor:s,count:function(){return this.eachAfter(a)},each:function(n,r){let e=-1;for(const t of this)n.call(r,t,++e,this);return this},eachAfter:function(n,r){for(var e,t,i,o=this,u=[o],a=[],f=-1;o=u.pop();)if(a.push(o),e=o.children)for(t=0,i=e.length;t<i;++t)u.push(e[t]);for(;o=a.pop();)n.call(r,o,++f,this);return this},eachBefore:function(n,r){for(var e,t,i=this,o=[i],u=-1;i=o.pop();)if(n.call(r,i,++u,this),e=i.children)for(t=e.length-1;t>=0;--t)o.push(e[t]);return this},find:function(n,r){let e=-1;for(const t of this)if(n.call(r,t,++e,this))return t},sum:function(n){return this.eachAfter((function(r){for(var e=+n(r.data)||0,t=r.children,i=t&&t.length;--i>=0;)e+=t[i].value;r.value=e}))},sort:function(n){return this.eachBefore((function(r){r.children&&r.children.sort(n)}))},path:function(n){for(var r=this,e=function(n,r){if(n===r)return n;var e=n.ancestors(),t=r.ancestors(),i=null;for(n=e.pop(),r=t.pop();n===r;)i=n,n=e.pop(),r=t.pop();return i}(r,n),t=[r];r!==e;)r=r.parent,t.push(r);for(var i=t.length;n!==e;)t.splice(i,0,n),n=n.parent;return t},ancestors:function(){for(var n=this,r=[n];n=n.parent;)r.push(n);return r},descendants:function(){return Array.from(this)},leaves:function(){var n=[];return this.eachBefore((function(r){r.children||n.push(r)})),n},links:function(){var n=this,r=[];return n.each((function(e){e!==n&&r.push({source:e.parent,target:e})})),r},copy:function(){return f(this).eachBefore(l)},[Symbol.iterator]:function*(){var n,r,e,t,i=this,o=[i];do{for(n=o.reverse(),o=[];i=n.pop();)if(yield i,r=i.children)for(e=0,t=r.length;e<t;++e)o.push(r[e])}while(o.length)}};const g=1664525,m=1013904223,w=4294967296;function _(){let n=1;return()=>(n=(g*n+m)%w)/w}function B(n){return M(n,_())}function M(n,r){for(var e,t,i=0,o=(n=function(n,r){let e,t,i=n.length;for(;i;)t=r()*i--|0,e=n[i],n[i]=n[t],n[t]=e;return n}(Array.from(n),r)).length,u=[];i<o;)e=n[i],t&&S(t,e)?++i:(t=O(u=z(u,e)),i=0);return t}function z(n,r){var e,t;if(b(r,n))return[r];for(e=0;e<n.length;++e)if(A(r,n[e])&&b(I(n[e],r),n))return[n[e],r];for(e=0;e<n.length-1;++e)for(t=e+1;t<n.length;++t)if(A(I(n[e],n[t]),r)&&A(I(n[e],r),n[t])&&A(I(n[t],r),n[e])&&b(q(n[e],n[t],r),n))return[n[e],n[t],r];throw new Error}function A(n,r){var e=n.r-r.r,t=r.x-n.x,i=r.y-n.y;return e<0||e*e<t*t+i*i}function S(n,r){var e=n.r-r.r+1e-9*Math.max(n.r,r.r,1),t=r.x-n.x,i=r.y-n.y;return e>0&&e*e>t*t+i*i}function b(n,r){for(var e=0;e<r.length;++e)if(!S(n,r[e]))return!1;return!0}function O(n){switch(n.length){case 1:return function(n){return{x:n.x,y:n.y,r:n.r}}(n[0]);case 2:return I(n[0],n[1]);case 3:return q(n[0],n[1],n[2])}}function I(n,r){var e=n.x,t=n.y,i=n.r,o=r.x,u=r.y,a=r.r,f=o-e,c=u-t,h=a-i,l=Math.sqrt(f*f+c*c);return{x:(e+o+f/l*h)/2,y:(t+u+c/l*h)/2,r:(l+i+a)/2}}function q(n,r,e){var t=n.x,i=n.y,o=n.r,u=r.x,a=r.y,f=r.r,c=e.x,h=e.y,l=e.r,p=t-u,s=t-c,d=i-a,y=i-h,v=f-o,x=l-o,g=t*t+i*i-o*o,m=g-u*u-a*a+f*f,w=g-c*c-h*h+l*l,_=s*d-p*y,B=(d*w-y*m)/(2*_)-t,M=(y*v-d*x)/_,z=(s*m-p*w)/(2*_)-i,A=(p*x-s*v)/_,S=M*M+A*A-1,b=2*(o+B*M+z*A),O=B*B+z*z-o*o,I=-(Math.abs(S)>1e-6?(b+Math.sqrt(b*b-4*S*O))/(2*S):O/b);return{x:t+B+M*I,y:i+z+A*I,r:I}}function E(n,r,e){var t,i,o,u,a=n.x-r.x,f=n.y-r.y,c=a*a+f*f;c?(i=r.r+e.r,i*=i,u=n.r+e.r,i>(u*=u)?(t=(c+u-i)/(2*c),o=Math.sqrt(Math.max(0,u/c-t*t)),e.x=n.x-t*a-o*f,e.y=n.y-t*f+o*a):(t=(c+i-u)/(2*c),o=Math.sqrt(Math.max(0,i/c-t*t)),e.x=r.x+t*a-o*f,e.y=r.y+t*f+o*a)):(e.x=r.x+e.r,e.y=r.y)}function P(n,r){var e=n.r+r.r-1e-6,t=r.x-n.x,i=r.y-n.y;return e>0&&e*e>t*t+i*i}function k(n){var r=n._,e=n.next._,t=r.r+e.r,i=(r.x*e.r+e.x*r.r)/t,o=(r.y*e.r+e.y*r.r)/t;return i*i+o*o}function N(n){this._=n,this.next=null,this.previous=null}function j(n,r){if(!(u=(e=n,n="object"==typeof e&&"length"in e?e:Array.from(e)).length))return 0;var e,t,i,o,u,a,f,c,h,l,p,s;if((t=n[0]).x=0,t.y=0,!(u>1))return t.r;if(i=n[1],t.x=-i.r,i.x=t.r,i.y=0,!(u>2))return t.r+i.r;E(i,t,o=n[2]),t=new N(t),i=new N(i),o=new N(o),t.next=o.previous=i,i.next=t.previous=o,o.next=i.previous=t;n:for(c=3;c<u;++c){E(t._,i._,o=n[c]),o=new N(o),h=i.next,l=t.previous,p=i._.r,s=t._.r;do{if(p<=s){if(P(h._,o._)){i=h,t.next=i,i.previous=t,--c;continue n}p+=h._.r,h=h.next}else{if(P(l._,o._)){(t=l).next=i,i.previous=t,--c;continue n}s+=l._.r,l=l.previous}}while(h!==l.next);for(o.previous=t,o.next=i,t.next=i.previous=i=o,a=k(t);(o=o.next)!==i;)(f=k(o))<a&&(t=o,a=f);i=t.next}for(t=[i._],o=i;(o=o.next)!==i;)t.push(o._);for(o=M(t,r),c=0;c<u;++c)(t=n[c]).x-=o.x,t.y-=o.y;return o.r}function D(n){return j(n,_()),n}function T(n){return Math.sqrt(n.value)}function $(){var n=null,r=1,e=1,t=v;function i(i){const o=_();return i.x=r/2,i.y=e/2,n?i.eachBefore(L(n)).eachAfter(R(t,.5,o)).eachBefore(C(1)):i.eachBefore(L(T)).eachAfter(R(v,1,o)).eachAfter(R(t,i.r/Math.min(r,e),o)).eachBefore(C(Math.min(r,e)/(2*i.r))),i}return i.radius=function(r){return arguments.length?(n=d(r),i):n},i.size=function(n){return arguments.length?(r=+n[0],e=+n[1],i):[r,e]},i.padding=function(n){return arguments.length?(t="function"==typeof n?n:x(+n),i):t},i}function L(n){return function(r){r.children||(r.r=Math.max(0,+n(r)||0))}}function R(n,r,e){return function(t){if(i=t.children){var i,o,u,a=i.length,f=n(t)*r||0;if(f)for(o=0;o<a;++o)i[o].r+=f;if(u=j(i,e),f)for(o=0;o<a;++o)i[o].r-=f;t.r=u+f}}}function C(n){return function(r){var e=r.parent;r.r*=n,e&&(r.x=e.x+n*r.x,r.y=e.y+n*r.y)}}function F(n){n.x0=Math.round(n.x0),n.y0=Math.round(n.y0),n.x1=Math.round(n.x1),n.y1=Math.round(n.y1)}function G(n,r,e,t,i){for(var o,u=n.children,a=-1,f=u.length,c=n.value&&(t-r)/n.value;++a<f;)(o=u[a]).y0=e,o.y1=i,o.x0=r,o.x1=r+=o.value*c}function U(){var n=1,r=1,e=0,t=!1;function i(i){var o=i.height+1;return i.x0=i.y0=e,i.x1=n,i.y1=r/o,i.eachBefore(function(n,r){return function(t){t.children&&G(t,t.x0,n*(t.depth+1)/r,t.x1,n*(t.depth+2)/r);var i=t.x0,o=t.y0,u=t.x1-e,a=t.y1-e;u<i&&(i=u=(i+u)/2),a<o&&(o=a=(o+a)/2),t.x0=i,t.y0=o,t.x1=u,t.y1=a}}(r,o)),t&&i.eachBefore(F),i}return i.round=function(n){return arguments.length?(t=!!n,i):t},i.size=function(e){return arguments.length?(n=+e[0],r=+e[1],i):[n,r]},i.padding=function(n){return arguments.length?(e=+n,i):e},i}var V={depth:-1},H={},J={};function K(n){return n.id}function Q(n){return n.parentId}function W(){var n,r=K,e=Q;function t(t){var i,o,u,a,f,c,h,l,d=Array.from(t),y=r,v=e,x=new Map;if(null!=n){const r=d.map(((r,e)=>function(n){let r=(n=`${n}`).length;return Y(n,r-1)&&!Y(n,r-2)&&(n=n.slice(0,-1)),"/"===n[0]?n:`/${n}`}(n(r,e,t)))),e=r.map(X),i=new Set(r).add("");for(const n of e)i.has(n)||(i.add(n),r.push(n),e.push(X(n)),d.push(J));y=(n,e)=>r[e],v=(n,r)=>e[r]}for(u=0,i=d.length;u<i;++u)o=d[u],c=d[u]=new s(o),null!=(h=y(o,u,t))&&(h+="")&&(l=c.id=h,x.set(l,x.has(l)?H:c)),null!=(h=v(o,u,t))&&(h+="")&&(c.parent=h);for(u=0;u<i;++u)if(h=(c=d[u]).parent){if(!(f=x.get(h)))throw new Error("missing: "+h);if(f===H)throw new Error("ambiguous: "+h);f.children?f.children.push(c):f.children=[c],c.parent=f}else{if(a)throw new Error("multiple roots");a=c}if(!a)throw new Error("no root");if(null!=n){for(;a.data===J&&1===a.children.length;)a=a.children[0],--i;for(let n=d.length-1;n>=0&&(c=d[n]).data===J;--n)c.data=null}if(a.parent=V,a.eachBefore((function(n){n.depth=n.parent.depth+1,--i})).eachBefore(p),a.parent=null,i>0)throw new Error("cycle");return a}return t.id=function(n){return arguments.length?(r=d(n),t):r},t.parentId=function(n){return arguments.length?(e=d(n),t):e},t.path=function(r){return arguments.length?(n=d(r),t):n},t}function X(n){let r=n.length;if(r<2)return"";for(;--r>1&&!Y(n,r););return n.slice(0,r)}function Y(n,r){if("/"===n[r]){let e=0;for(;r>0&&"\\"===n[--r];)++e;if(!(1&e))return!0}return!1}function Z(n,r){return n.parent===r.parent?1:2}function nn(n){var r=n.children;return r?r[0]:n.t}function rn(n){var r=n.children;return r?r[r.length-1]:n.t}function en(n,r,e){var t=e/(r.i-n.i);r.c-=t,r.s+=e,n.c+=t,r.z+=e,r.m+=e}function tn(n,r,e){return n.a.parent===r.parent?n.a:e}function on(n,r){this._=n,this.parent=null,this.children=null,this.A=null,this.a=this,this.z=0,this.m=0,this.c=0,this.s=0,this.t=null,this.i=r}function un(){var n=Z,r=1,e=1,t=null;function i(i){var f=function(n){for(var r,e,t,i,o,u=new on(n,0),a=[u];r=a.pop();)if(t=r._.children)for(r.children=new Array(o=t.length),i=o-1;i>=0;--i)a.push(e=r.children[i]=new on(t[i],i)),e.parent=r;return(u.parent=new on(null,0)).children=[u],u}(i);if(f.eachAfter(o),f.parent.m=-f.z,f.eachBefore(u),t)i.eachBefore(a);else{var c=i,h=i,l=i;i.eachBefore((function(n){n.x<c.x&&(c=n),n.x>h.x&&(h=n),n.depth>l.depth&&(l=n)}));var p=c===h?1:n(c,h)/2,s=p-c.x,d=r/(h.x+p+s),y=e/(l.depth||1);i.eachBefore((function(n){n.x=(n.x+s)*d,n.y=n.depth*y}))}return i}function o(r){var e=r.children,t=r.parent.children,i=r.i?t[r.i-1]:null;if(e){!function(n){for(var r,e=0,t=0,i=n.children,o=i.length;--o>=0;)(r=i[o]).z+=e,r.m+=e,e+=r.s+(t+=r.c)}(r);var o=(e[0].z+e[e.length-1].z)/2;i?(r.z=i.z+n(r._,i._),r.m=r.z-o):r.z=o}else i&&(r.z=i.z+n(r._,i._));r.parent.A=function(r,e,t){if(e){for(var i,o=r,u=r,a=e,f=o.parent.children[0],c=o.m,h=u.m,l=a.m,p=f.m;a=rn(a),o=nn(o),a&&o;)f=nn(f),(u=rn(u)).a=r,(i=a.z+l-o.z-c+n(a._,o._))>0&&(en(tn(a,r,t),r,i),c+=i,h+=i),l+=a.m,c+=o.m,p+=f.m,h+=u.m;a&&!rn(u)&&(u.t=a,u.m+=l-h),o&&!nn(f)&&(f.t=o,f.m+=c-p,t=r)}return t}(r,i,r.parent.A||t[0])}function u(n){n._.x=n.z+n.parent.m,n.m+=n.parent.m}function a(n){n.x*=r,n.y=n.depth*e}return i.separation=function(r){return arguments.length?(n=r,i):n},i.size=function(n){return arguments.length?(t=!1,r=+n[0],e=+n[1],i):t?null:[r,e]},i.nodeSize=function(n){return arguments.length?(t=!0,r=+n[0],e=+n[1],i):t?[r,e]:null},i}function an(n,r,e,t,i){for(var o,u=n.children,a=-1,f=u.length,c=n.value&&(i-e)/n.value;++a<f;)(o=u[a]).x0=r,o.x1=t,o.y0=e,o.y1=e+=o.value*c}on.prototype=Object.create(s.prototype);var fn=(1+Math.sqrt(5))/2;function cn(n,r,e,t,i,o){for(var u,a,f,c,h,l,p,s,d,y,v,x=[],g=r.children,m=0,w=0,_=g.length,B=r.value;m<_;){f=i-e,c=o-t;do{h=g[w++].value}while(!h&&w<_);for(l=p=h,v=h*h*(y=Math.max(c/f,f/c)/(B*n)),d=Math.max(p/v,v/l);w<_;++w){if(h+=a=g[w].value,a<l&&(l=a),a>p&&(p=a),v=h*h*y,(s=Math.max(p/v,v/l))>d){h-=a;break}d=s}x.push(u={value:h,dice:f<c,children:g.slice(m,w)}),u.dice?G(u,e,t,i,B?t+=c*h/B:o):an(u,e,t,B?e+=f*h/B:i,o),B-=h,m=w}return x}const hn=function n(r){function e(n,e,t,i,o){cn(r,n,e,t,i,o)}return e.ratio=function(r){return n((r=+r)>1?r:1)},e}(fn);function ln(){var n=hn,r=!1,e=1,t=1,i=[0],o=v,u=v,a=v,f=v,c=v;function h(n){return n.x0=n.y0=0,n.x1=e,n.y1=t,n.eachBefore(l),i=[0],r&&n.eachBefore(F),n}function l(r){var e=i[r.depth],t=r.x0+e,h=r.y0+e,l=r.x1-e,p=r.y1-e;l<t&&(t=l=(t+l)/2),p<h&&(h=p=(h+p)/2),r.x0=t,r.y0=h,r.x1=l,r.y1=p,r.children&&(e=i[r.depth+1]=o(r)/2,t+=c(r)-e,h+=u(r)-e,(l-=a(r)-e)<t&&(t=l=(t+l)/2),(p-=f(r)-e)<h&&(h=p=(h+p)/2),n(r,t,h,l,p))}return h.round=function(n){return arguments.length?(r=!!n,h):r},h.size=function(n){return arguments.length?(e=+n[0],t=+n[1],h):[e,t]},h.tile=function(r){return arguments.length?(n=y(r),h):n},h.padding=function(n){return arguments.length?h.paddingInner(n).paddingOuter(n):h.paddingInner()},h.paddingInner=function(n){return arguments.length?(o="function"==typeof n?n:x(+n),h):o},h.paddingOuter=function(n){return arguments.length?h.paddingTop(n).paddingRight(n).paddingBottom(n).paddingLeft(n):h.paddingTop()},h.paddingTop=function(n){return arguments.length?(u="function"==typeof n?n:x(+n),h):u},h.paddingRight=function(n){return arguments.length?(a="function"==typeof n?n:x(+n),h):a},h.paddingBottom=function(n){return arguments.length?(f="function"==typeof n?n:x(+n),h):f},h.paddingLeft=function(n){return arguments.length?(c="function"==typeof n?n:x(+n),h):c},h}function pn(n,r,e,t,i){var o,u,a=n.children,f=a.length,c=new Array(f+1);for(c[0]=u=o=0;o<f;++o)c[o+1]=u+=a[o].value;!function n(r,e,t,i,o,u,f){if(r>=e-1){var h=a[r];return h.x0=i,h.y0=o,h.x1=u,void(h.y1=f)}for(var l=c[r],p=t/2+l,s=r+1,d=e-1;s<d;){var y=s+d>>>1;c[y]<p?s=y+1:d=y}p-c[s-1]<c[s]-p&&r+1<s&&--s;var v=c[s]-l,x=t-v;if(u-i>f-o){var g=t?(i*x+u*v)/t:u;n(r,s,v,i,o,g,f),n(s,e,x,g,o,u,f)}else{var m=t?(o*x+f*v)/t:f;n(r,s,v,i,o,u,m),n(s,e,x,i,m,u,f)}}(0,f,n.value,r,e,t,i)}function sn(n,r,e,t,i){(1&n.depth?an:G)(n,r,e,t,i)}const dn=function n(r){function e(n,e,t,i,o){if((u=n._squarify)&&u.ratio===r)for(var u,a,f,c,h,l=-1,p=u.length,s=n.value;++l<p;){for(f=(a=u[l]).children,c=a.value=0,h=f.length;c<h;++c)a.value+=f[c].value;a.dice?G(a,e,t,i,s?t+=(o-t)*a.value/s:o):an(a,e,t,s?e+=(i-e)*a.value/s:i,o),s-=a.value}else n._squarify=u=cn(r,n,e,t,i,o),u.ratio=r}return e.ratio=function(r){return n((r=+r)>1?r:1)},e}(fn)}},r={};function e(t){var i=r[t];if(void 0!==i)return i.exports;var o=r[t]={exports:{}};return n[t](o,o.exports,e),o.exports}e.d=(n,r)=>{for(var t in r)e.o(r,t)&&!e.o(n,t)&&Object.defineProperty(n,t,{enumerable:!0,get:r[t]})},e.o=(n,r)=>Object.prototype.hasOwnProperty.call(n,r),e.r=n=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(n,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(n,"__esModule",{value:!0})};const{hierarchy:t,tree:i}=e(188),o=t({name:"Birthday Party Tasks",reasoning:"",children:[{name:"Design the Garden Layout",reasoning:"A well-thought-out plan is needed...",children:[{name:"NO SUBDIVISION",reasoning:"Garden layout design complexity is unknown."}]},{name:"Choose Plants",reasoning:"Selecting the right plants is crucial...",children:[{name:"NO SUBDIVISION",reasoning:"No time frame or plant type specified."}]},{name:"Prepare the Soil",reasoning:"Proper soil preparation ensures nutrient-rich growth.",children:[]}]});i().size([400,200])(o),o.descendants().forEach((n=>{console.log(`Node: ${n.data.name}`),console.log(`  Depth: ${n.depth}`),console.log(`  Coordinates (x, y): (${n.x.toFixed(2)}, ${n.y.toFixed(2)})`)}))})();